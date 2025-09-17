import argparse
import os
import os.path as osp
import pdb
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
import tqdm
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys
from echoclip.runner.data import EchoDataModule,Echo
from echoclip.runner.data import EchoDataModule

from echoclip.runner.runner import Runner
from echoclip.utils.logging import get_logger, setup_file_handle_for_all_logger


logger = get_logger(__name__)



def main(cfg: DictConfig):
    pl.seed_everything(cfg.runner_cfg.seed, True)
    output_dir = Path(cfg.runner_cfg.output_dir)
    setup_file_handle_for_all_logger(str(output_dir / "run.log"))

    callbacks = load_callbacks(output_dir)
    loggers = load_loggers(output_dir)

    # deterministic = True
    deterministic = False
    logger.info(f"`deterministic` flag: {deterministic}")

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        deterministic=deterministic,
        **OmegaConf.to_container(cfg.trainer_cfg),
    )

    if cfg.trainer_cfg.fast_dev_run is True:
        from IPython.core.debugger import set_trace

        set_trace()

    runner = None
    regression_datamodule = EchoDataModule(**OmegaConf.to_container(cfg.data_cfg))


    # Training
    if not cfg.test_only:
        runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))

        logger.info("Start training.")
        trainer.fit(model=runner, datamodule=regression_datamodule)
        trainer.test(model=runner, datamodule=regression_datamodule)

        logger.info("End training.")

    # Testing
    ckpt_paths = list((output_dir / "ckpts").glob("*.ckpt"))
    if len(ckpt_paths) == 0:
        logger.info("zero shot")
        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=regression_datamodule)
        logger.info(f"End zero shot.")

    for ckpt_path in ckpt_paths:
        logger.info(f"Start testing ckpt: {ckpt_path}.")

        for k in cfg.runner_cfg.load_weights_cfg.keys():
            cfg.runner_cfg.load_weights_cfg[k] = None
        cfg.runner_cfg.ckpt_path = str(ckpt_path)

        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))

        runner = runner.load_from_checkpoint(str(ckpt_path), **OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model=runner, datamodule=regression_datamodule)

        logger.info(f"End testing ckpt: {ckpt_path}.")


def load_loggers(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "tb_logger").mkdir(exist_ok=True, parents=True)
    (output_dir / "csv_logger").mkdir(exist_ok=True, parents=True)
    loggers = []
    loggers.append(
        pl_loggers.CSVLogger(
            str(output_dir),
            name="csv_logger",
        )
    )

    return loggers


def load_callbacks(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "ckpts").mkdir(exist_ok=True, parents=True)

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val_mae_exp_metric",
            dirpath=str(output_dir / "ckpts"),
            filename="{epoch:02d}-{val_mae_exp_metric:.4f}",
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        )

    )
    return callbacks


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    return output_dir


def get_version(path: Path):
    versions = path.glob("version_*")
    return len(list(versions))


def parse_cfg(args, instantialize_output_dir=True):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed if args.seed is not None else cfg.runner_cfg.seed
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg



def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.
    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        np.random.seed(0)
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    # for (x, *_) in tqdm.tqdm(dataloader):
    for (x,_,*_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = parse_cfg(args, instantialize_output_dir=True)

    logger.info("Start.")
    main(cfg)
    logger.info("End.")
