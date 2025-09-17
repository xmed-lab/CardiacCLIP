import math
import os
import os.path as osp
import random
from collections import defaultdict

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from echoclip.utils.logging import get_logger, print_log
import pandas
import numpy as np
import skimage.draw
import torchvision
import utils
import collections

from .utils import get_transforms
from torchvision import transforms


import torch.nn as nn
import torchvision.transforms as transforms
import nibabel as nib
import cv2
import SimpleITK as sitk


logger = get_logger(__name__)
print = lambda x: print_log(x, logger=logger)

import sys


class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images_root,
        val_images_root,
        test_images_root,
        train_data_file,
        val_data_file,
        test_data_file,
        transforms_cfg=None,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        few_shot=None,
        label_distributed_shift=None,
        use_long_tail=False
    ):
        super().__init__()
        train_transforms, eval_transforms = get_transforms(**transforms_cfg)

        self.train_set = RegressionDataset(train_images_root, train_data_file, train_transforms)
        self.val_set = RegressionDataset(val_images_root, val_data_file, eval_transforms)
        self.test_set = RegressionDataset(test_images_root, test_data_file, eval_transforms)

        self.train_set.generate_fewshot_dataset(**few_shot)
        self.train_set.generate_distribution_shifted_dataset(**label_distributed_shift)
        if use_long_tail:
            self.val_set.generate_long_tail()
            self.test_set.generate_long_tail()

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)


class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms=None):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms

        with open(data_file) as fin:
            for line in fin:
                # image_file, image_label = line.split()
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                self.labels.append([int(label) for label in labels])
                self.images_file.append(image_file)

        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")

    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)

        if img.mode == "L":
            img = img.convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def generate_long_tail(self):
        images_file_new, labels_new = [], []
        len_before = len(self.labels)
        for index in range(len_before):
            img_file, target_list = self.images_file[index], self.labels[index]
            if "val" in self.name or "test" in self.name:
                target = target_list[len(target_list) // 2]
            else:
                target = random.choice(target_list)
            if target >= 50:
                images_file_new.append(img_file)
                labels_new.append(target_list)
        
        self.images_file = images_file_new
        self.labels = labels_new
        len_after = len(self.labels)
        logger.info(f"generate long tail dataset, the change of # of samples: {len_before} -> {len_after}.")

    def get_label_dist(self, target):
        label_dist = [self.normal_sampling(int(target), i, std=self.std) for i in range(self.n_cls)]
        label_dist = [i if i > 1e-15 else 1e-15 for i in label_dist]
        label_dist = torch.Tensor(label_dist)

        return label_dist

    def __len__(self):
        return len(self.labels)

    def split_dataset_by_label(self):
        output = defaultdict(list)
        for img, label in zip(self.images_file, self.labels):
            target = label[len(label) // 2]
            output[target].append(img)
        return output

    def generate_fewshot_dataset(self, num_shots=-1, repeat=False):
        if num_shots <= 0:
            print("not generate few-shot dataset: num_shots<=0")
            return

        output = self.split_dataset_by_label()

        print("generate few-shot dataset")
        print("clear full dataset: images_file & labels")
        self._images_file = self.images_file
        self._labels = self.labels

        self.images_file = []
        self.labels = []
        print(
            f"build few_shot: num labels: {len(output.keys())}, {list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )
        for label, imgs_ls in output.items():
            # self.images_file.extend(imgs_ls)
            # self.labels.extend([label] * len(imgs_ls))

            if len(imgs_ls) >= num_shots:
                sampled_imgs_ls = random.sample(imgs_ls, num_shots)
            else:
                print(f"not enough: class-{label}: {len(imgs_ls)}")
                if repeat:
                    sampled_imgs_ls = random.choices(imgs_ls, k=num_shots)
                else:
                    sampled_imgs_ls = imgs_ls

            self.images_file.extend(sampled_imgs_ls)
            self.labels.extend([[label]] * len(sampled_imgs_ls))
        assert len(self.images_file) == len(self.labels), f"{len(self.images_file)} != {len(self.lables)}"
        print(f"len of few shot dataset: {len(self.images_file)}")

    def generate_distribution_shifted_dataset(self, num_topk_scaled_class=-1, scale_factor=0.3):
        if num_topk_scaled_class <= 0:
            print("not generate distribution shifted dataset: num_topk_scaled_class<=1")
            return
        if scale_factor == 1.0:
            print("not generate distribution shifted dataset: scale_factor=1.0")
            return
        assert scale_factor > 0 and scale_factor < 1.0

        output = self.split_dataset_by_label()

        print("generate distribution shifted dataset")
        print("clear full dataset: images_file & labels")
        self._images_file = self.images_file
        self._labels = self.labels

        self.images_file = []
        self.labels = []

        print(
            f"build distribution shifed: num labels: {len(output.keys())}, {list(output.keys())[:5]}, ..., {list(output.keys())[-5:]}"
        )

        num_samples_per_label = [[k, len(v)] for k, v in output.items()]
        num_samples_per_label.sort(key=lambda x: x[1], reverse=True)

        for idx, label_cnt in enumerate(num_samples_per_label):
            if idx < num_topk_scaled_class:
                imgs_ls = output[label_cnt[0]]
                sampled_imgs_ls = random.sample(imgs_ls, max(int(len(imgs_ls) * scale_factor), 1))
            else:
                sampled_imgs_ls = output[label_cnt[0]]

            self.images_file.extend(sampled_imgs_ls)
            self.labels.extend([[label_cnt[0]]] * len(sampled_imgs_ls))

        assert len(self.images_file) == len(self.labels), f"{len(self.images_file)} != {len(self.lables)}"
        print(f"len of distribution shifed dataset: {len(self.images_file)}")

    @staticmethod
    def normal_sampling(mean, label_k, std=2):
        return math.exp(-((label_k - mean) ** 2) / (2 * std**2)) / (math.sqrt(2 * math.pi) * std)



class EchoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        transforms_cfg=None,
    ):
        super().__init__()



        self.train_set = Echo(split="train", pad=12)
        self.val_set = Echo(split="val")
        self.test_set = Echo(split="test")


        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)




class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=48, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        if root is None:
            # root = "/home/ydubf/download/EchoNet-Dynamic"
            root = "/nfs/usrhome/ydubf/download/EchoNet-Dynamic"

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
# with normal few-shot sample for training, half val and normal test set
            with open(os.path.join(self.root, "few-shot/FileList_1_shot_mediam.csv")) as f:
            # with open(os.path.join(self.root, "few-shot/FileList_2_shot_mediam.csv")) as f:
            # with open(os.path.join(self.root, "few-shot/FileList_4_shot_mediam.csv")) as f:
            # with open(os.path.join(self.root, "few-shot/FileList_8_shot_mediam.csv")) as f:



                data = pandas.read_csv(f)

            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = utils.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]
        
        # upsampling from 112 to 224
        # c, l, h, w = video.shape

        

        return video, target

    def __len__(self):
        return len(self.fnames)


    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)



def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)