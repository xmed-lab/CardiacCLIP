import torch
import numpy as np
from pathlib import Path
import cv2
import re
import os
import typing
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
import sys


LVEF_fine_prompts = ["Severely Reduced LVEF (<30%). General Appearance: Marked impairment in heart function; Left Ventricle: Poor contractility, potential dilation; Ejection Fraction: Severe systolic dysfunction; Wall Motion: Extensive hypokinesia or akinesia; possible dyskinesia.",
                    "Moderately Reduced LVEF (30-44%). General Appearance: Noticeable reduction in heart's pumping ability; Left Ventricle: Significantly decreased contractility; Ejection Fraction: Indicates moderate systolic dysfunction; Wall Motion: Areas of hypokinesia or akinesia may be observed.",
                    "Mildly Reduced LVEF (45-54%). General Appearance: Slightly reduced pumping efficiency; Left Ventricle: Contractions are somewhat weaker; Ejection Fraction: Borderline systolic function; Wall Motion: Mild hypokinesia may be present.",
                    "Normal LVEF (55-70%). General Appearance: The heart functions efficiently; Left Ventricle: Shows strong and coordinated contractions; Ejection Fraction: Indicates normal systolic function; Wall Motion: Uniform and vigorous.",
                    "Hyperdynamic LVEF (>70%). General Appearance: The heart appears to pump vigorously; Left Ventricle: Exhibits very strong contractions; Ejection Fraction: Higher than normal, suggesting increased systolic function; Wall Motion: Hyperkinesia (excessive motion) is evident."]


LVEF_fine_prompts_new = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE Severely Reduced LVEF (<30%). General Appearance: Marked impairment in heart function; Left Ventricle: Poor contractility, potential dilation; Ejection Fraction: Severe systolic dysfunction; Wall Motion: Extensive hypokinesia or akinesia; possible dyskinesia.",
                    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE Moderately Reduced LVEF (30-44%). General Appearance: Noticeable reduction in heart's pumping ability; Left Ventricle: Significantly decreased contractility; Ejection Fraction: Indicates moderate systolic dysfunction; Wall Motion: Areas of hypokinesia or akinesia may be observed.",
                    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE Mildly Reduced LVEF (45-54%). General Appearance: Slightly reduced pumping efficiency; Left Ventricle: Contractions are somewhat weaker; Ejection Fraction: Borderline systolic function; Wall Motion: Mild hypokinesia may be present.",
                    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE Normal LVEF (55-70%). General Appearance: The heart functions efficiently; Left Ventricle: Shows strong and coordinated contractions; Ejection Fraction: Indicates normal systolic function; Wall Motion: Uniform and vigorous.",
                    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE Hyperdynamic LVEF (>70%). General Appearance: The heart appears to pump vigorously; Left Ventricle: Exhibits very strong contractions; Ejection Fraction: Higher than normal, suggesting increased systolic function; Wall Motion: Hyperkinesia (excessive motion) is evident."]


LVEF_fine_prompt0 = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>%. Severely Reduced LVEF (<30%). General Appearance: Marked impairment in heart function; Left Ventricle: Poor contractility, potential dilation; Ejection Fraction: Severe systolic dysfunction; Wall Motion: Extensive hypokinesia or akinesia; possible dyskinesia."]

LVEF_fine_prompt1 = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>%. Moderately Reduced LVEF (30-44%). General Appearance: Noticeable reduction in heart's pumping ability; Left Ventricle: Significantly decreased contractility; Ejection Fraction: Indicates moderate systolic dysfunction; Wall Motion: Areas of hypokinesia or akinesia may be observed."]

LVEF_fine_prompt2 = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>%. Mildly Reduced LVEF (45-54%). General Appearance: Slightly reduced pumping efficiency; Left Ventricle: Contractions are somewhat weaker; Ejection Fraction: Borderline systolic function; Wall Motion: Mild hypokinesia may be present."]

LVEF_fine_prompt3 = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>%. Normal LVEF (55-70%). General Appearance: The heart functions efficiently; Left Ventricle: Shows strong and coordinated contractions; Ejection Fraction: Indicates normal systolic function; Wall Motion: Uniform and vigorous."]

LVEF_fine_prompt4 = ["THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>%. Hyperdynamic LVEF (>70%). General Appearance: The heart appears to pump vigorously; Left Ventricle: Exhibits very strong contractions; Ejection Fraction: Higher than normal, suggesting increased systolic function; Wall Motion: Hyperkinesia (excessive motion) is evident."]



zero_shot_prompts = {
    "ejection_fraction": [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ],
    "pacemaker": [
        "ECHO DENSITY IN RIGHT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
        "ECHO DENSITY IN RIGHT ATRIUM SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
    ],
    "impella": [
        "AN IMPELLA CATHETER IS SEEN AND THE INLET AREA IS 4.0CM FROM THE AORTIC VALVE AND DOES NOT INTERFERE WITH NEIGHBORING STRUCTURES, CONSISTENT WITH CORRECT IMPELLA POSITIONING. THERE IS DENSE TURBULENT COLOR FLOW ABOVE THE AORTIC VALVE, CONSISTENT WITH CORRECT OUTFLOW AREA POSITION ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND IS TOO CLOSE TO OR ENTANGLED IN THE PAPILLARY MUSCLE AND SUBANNULAR STRUCTURES SURROUNDING THE MITRAL VALVE; REPOSITIONING RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN, HOWEVER THE INLET AREA APPEARS TO BE IN THE AORTA OR NEAR THE AORTIC VALVE; REPOSITIONING IS RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND EXTENDS TOO FAR INTO THE LEFT VENTRICLE; REPOSITIONING RECOMMENDED ",
    ],
    "normal_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA SHOWS A NORMAL RESPIRATORY COLLAPSE CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (3MMHG). ",
    ],
    "elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES LESS THAN 50% COLLAPSE CONSISTENT WITH ELEVATED RIGHT ATRIAL PRESSURE (8MMHG). ",
    ],
    "significantly_elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES NO INSPIRATORY COLLAPSE, CONSISTENT WITH SIGNIFICANTLY ELEVATED RIGHT ATRIAL PRESSURE (>15MMHG). ",
    ],
    "pulmonary_artery_pressure": [
        "ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG. ",
        "ESTIMATED PA PRESSURE IS <#>MMHG. ",
        "PA PEAK PRESSURE IS <#>MMHG. ",
    ],
    "severe_left_ventricle_dilation": [
        "SEVERE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "SEVERE DILATED LEFT VENTRICLE BY VOLUME. ",
        "SEVERE DILATED LEFT VENTRICLE. ",
    ],
    "moderate_left_ventricle_dilation": [
        "MODERATE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MODERATE DILATED LEFT VENTRICLE BY VOLUME. ",
        "MODERATE DILATED LEFT VENTRICLE. ",
    ],
    "mild_left_ventricle_dilation": [
        "MILD DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MILD DILATED LEFT VENTRICLE BY VOLUME. ",
        "MILD DILATED LEFT VENTRICLE. ",
    ],
    "severe_right_ventricle_size": ["SEVERE DILATED RIGHT VENTRICLE. "],
    "moderate_right_ventricle_size": ["MODERATE DILATED RIGHT VENTRICLE. "],
    "mild_right_ventricle_size": ["MILD DILATED RIGHT VENTRICLE. "],
    "severe_left_atrium_size": ["SEVERE DILATED LEFT ATRIUM. "],
    "moderate_left_atrium_size": ["MODERATE DILATED LEFT ATRIUM. "],
    "mild_left_atrium_size": ["MILD DILATED LEFT ATRIUM. "],
    "severe_right_atrium_size": ["SEVERE DILATED RIGHT ATRIUM. "],
    "moderate_right_atrium_size": ["MODERATE DILATED RIGHT ATRIUM. "],
    "mild_right_atrium_size": ["MILD DILATED RIGHT ATRIUM. "],
    "tavr": [
        "A BIOPROSTHETIC STENT-VALVE IS PRESENT IN THE AORTIC POSITION. ",
    ],
    "mitraclip": [
        "TWO MITRACLIPS ARE SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
        "TWO MITRACLIPS ARE NOW PRESENT ON THE ANTERIOR AND POSTERIOR MITRAL VALVE LEAFLETS. ",
        "ONE MITRACLIP IS SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
    ],
}



def compute_regression_metric( 
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    logits: torch.Tensor,
    
):

    prob = F.softmax(logits, dim=-1)
    prompt_values = [15,37,50,62,85]
    prompt_values = torch.tensor(prompt_values).to(logits.device)
    pred_frame = prob * prompt_values
    pred_video = torch.sum(pred_frame,dim=-1)

    return pred_video




def crop_and_scale(img, res=(640, 480), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)

    return img


def read_avi(p: Path, res=None):
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


## TEXT CLEANING UTILS

removables = re.compile(r"\^|CRLF|‡")

in_text_periods = re.compile(r"(?<=\D)\.|\.(?=\D)")
square_brackets = re.compile(r"[\[\]]")
multi_whitespace = re.compile(r"\s+")
multi_period = re.compile(r"\.+")

select_was = re.compile(r"(?<=\b)WAS(?=\b)")
select_were = re.compile(r"(?<=\b)WERE(?=\b)")
select_and_or = re.compile(r"(?<=\b)AND/OR(?=\b)")
select_normally = re.compile(r"NORMALLY")
select_mildly = re.compile(r"MILDLY")
select_moderately = re.compile(r"MODERATELY")
select_severely = re.compile(r"SEVERELY")
select_pa = re.compile(r"PULMONARY ARTERY")
select_icd_codes = re.compile(r"[A-Z](\d+\.\d*\b)")
select_slash_dates = re.compile(r"\d{2}/\d{2}/\d{4}")
select_dot_dates = re.compile(r"\d{2}\.\d{2}\.\d{4}")

space_before_unit = re.compile(r"\s+(MMHG|MM|CM|%)")
space_period = re.compile(r"\s\.")

space_plus_space = re.compile(r"\s\+\s")
verbose_pressure = re.compile(r"\+CVPMMHG")
add_period = [
    r"THE PEAK TRANSAORTIC GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSAORTIC GRADIENT IS <#>MMHG",
    r"LV EJECTION FRACTION IS <#>%",
    r"ESTIMATED PA PRESSURE IS <#>MMHG",
    r"RESTING SEGMENTAL WALL MOTION ANALYSIS",
    r"THE IVC DIAMETER IS <#>MM",
    r"EST RV/RA PRESSURE GRADIENT IS <#>MMHG",
    r"ESTIMATED PEAK RVSP IS <#>MMHG",
    r"HEART FAILURE, UNSPECIFIED",
    r"CHEST PAIN, UNSPECIFIED",
    r"SINUS OF VALSALVA: <#>CM",
    r"THE PEAK TRANSMITRAL GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSMITRAL GRADIENT IS <#>MMHG",
    r"ASCENDING AORTA <#>CM",
    r"ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG",
    r"ICD_CODE SHORTNESS BREATH",
    r"ICD_CODE ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"SHORTNESS BREATH",
    r"ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"THE LEFT ATRIAL APPENDAGE IS NORMAL IN APPEARANCE WITH NO EVIDENCE OF THROMBUS",
]

select_number = r"(?:\d+\.?\d*)"

add_period = [re.escape(a).replace(re.escape("<#>"), select_number) for a in add_period]
add_period = [f"(?:{a})(?!\.)" for a in add_period]
add_period = "|".join(add_period)
add_period = f"({add_period})"
# print(f"{add_period[:50]} ... {add_period[-50:]}")
add_period = re.compile(add_period)


def clean_text(text):
    if len(text) > 1:
        text = text.upper()
        text = text.strip()
        text = text.replace("`", "'")
        text = removables.sub("", text)

        text = in_text_periods.sub(". ", text)
        text = square_brackets.sub("", text)

        text = select_was.sub("IS", text)
        text = select_were.sub("ARE", text)
        text = select_and_or.sub("AND", text)
        text = select_normally.sub("NORMAL", text)
        text = select_mildly.sub("MILD", text)
        text = select_moderately.sub("MODERATE", text)
        text = select_severely.sub("SEVERE", text)
        text = select_pa.sub("PA", text)
        text = select_slash_dates.sub("", text)
        text = select_dot_dates.sub("", text)
        text = select_icd_codes.sub("", text)

        text = space_before_unit.sub(r"\1", text)
        text = space_period.sub(".", text)
        text = multi_whitespace.sub(" ", text)

        text = space_plus_space.sub("+", text)
        text = verbose_pressure.sub("MMHG", text)

        text = text.strip()
        text = text + " "

        text = add_period.sub(r"\1.", text)
        text = multi_period.sub(".", text)

    return text


select_severity = "|".join(
    ["MODERATE/SEVERE", "MILD/MODERATE", "MILD", "MODERATE", "SEVERE", "VERY SEVERE"]
)
select_severity = f"((?<![A-Za-z])(?:{select_severity}))"
select_number = r"(\d+\.?\d*)"

select_variable = "|".join([select_number, select_severity])
# print(select_variable)
select_variable = re.compile(select_variable)


def extract_variables(string, replace_with="<#>"):
    matches = select_variable.findall(string)
    variables = []
    for match in matches:
        for variable in match:
            if not len(variable) == 0:
                variables.append(variable)
    variables_replaced = select_variable.sub(replace_with, string)
    return variables, variables_replaced



"""Utility functions for videos, plotting and computing performance metrics."""



def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, _, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in array.transpose((1, 2, 3, 0)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)


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
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def bootstrap(a, b, func, samples=10000):
    """Computes a bootstrapped confidence intervals for ``func(a, b)''.

    Args:
        a (array_like): first argument to `func`.
        b (array_like): second argument to `func`.
        func (callable): Function to compute confidence intervals for.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int, optional): Number of samples to compute.
            Defaults to 10000.

    Returns:
       A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


__all__ = ["video", "segmentation", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]
