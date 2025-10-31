# CardiacCLIP

This repository contains PyTorch implementation of ["CardiacCLIP: Video-based CLIP Adaptation for LVEF Prediction in a Few-shot Manner" (MICCAI 2025)](https://arxiv.org/abs/2509.17065).

Created by [Du Yao](https://scholar.google.com.hk/citations?user=8krbrWsAAAAJ&hl=zh-CN), [Guo Jiarong](https://scholar.google.com.hk/citations?hl=zh-CN&user=IT5sfsYAAAAJ&inst=1381320739207392350), [Li Xiaomeng](https://xmengli.github.io/)\*


## Overview of CardiacCLIP

CardiacCLIP is a novel adaptation of CLIP models for few-shot echocardiogram video analysis, capturing crucial temporal dynamics and localized cardiac structures essential for accurate diagnosis.

![intro](figs_CardiacCLIP/MIL_CLIP_1.png)


### 🔑 Key Idea

- **Multi-Frame Learning (MFL)**  
  An attention-based aggregation mechanism that **prioritizes diagnostically relevant frames** instead of simple averaging.  

- **EchoZoom**  
  A multi-scale input representation strategy that **enhances modeling of fine-grained cardiac structures**.  


The CardiacCLIP codebase is largely built upon [NumCLIP](https://github.com/xmed-lab/NumCLIP), sharing a similar overall architecture. To ease the difficulty of direct regression learning (LVEF prediction), we adopt a coarse-to-fine pipeline, where a classification stage is followed by a regression refinement step. For more details on this design, please refer to our ECCV paper on [NumCLIP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11339.pdf).

## Training & Evaluation


1. Change the dataset path in `/echoclip/runner/data.py` (around line 330).  Please download the [EchoNet-Dynamic Dataset](https://echonet.github.io/dynamic/)  first.



3. Run the training script:

```bash
sh scripts/run.sh
```

Results and logs will be saved in the results/ and wandb/ folders.


## One More Thing

Our project serves as a unified codebase for fine-tuning echocardiogram foundation models — such as [EchoCLIP](https://www.nature.com/articles/s41591-024-02959-y), [EchoPrime](https://arxiv.org/abs/2410.09704), and [PanEcho](https://jamanetwork.com/journals/jama/article-abstract/2835630) — supporting both full-model tuning and parameter-efficient finetuning approaches (e.g., [CoOp](https://github.com/KaiyangZhou/CoOp)) in a fast and modular manner.

You can simply define and load each model by initializing the corresponding encoder and loading its pretrained weights, as shown below.
```python
# Initialize and load EchoPrime video encoder
self.prime_encoder = models.video.mvit_v2_s()
device = torch.device("cuda")
checkpoint = torch.load("/home/ydubf/model_data/weights/echo_prime_encoder.pt", map_location=device)
self.prime_encoder.head[-1] = torch.nn.Linear(self.prime_encoder.head[-1].in_features, 512)
self.prime_encoder.load_state_dict(checkpoint)
self.prime_encoder.to(device)

# Initialize and load EchoPrime text encoder
self.prime_text_encoder = EchoPrimeTextEncoder()
checkpoint = torch.load("/home/ydubf/EchoPrime/model_data/weights/echo_prime_text_encoder.pt", map_location=device)
self.prime_text_encoder.load_state_dict(checkpoint)
self.prime_text_encoder.to(device)
```

Relevant Papers and Projects:

1. [EchoCLIP: Vision–language foundation model for echocardiogram interpretation](https://github.com/echonet/echo_CLIP)
2. [EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation ](https://github.com/echonet/EchoPrime)
3. [PanEcho: Complete AI-Enabled Echocardiography Interpretation With Multitask Deep Learning](https://github.com/CarDS-Yale/PanEcho)


## Citation

If you find this repository useful, please cite our work:

```
@inproceedings{du2025cardiacclip,
  title={CardiacCLIP: Video-Based CLIP Adaptation for LVEF Prediction in a Few-Shot Manner},
  author={Du, Yao and Guo, Jiarong and Li, Xiaomeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={46--56},
  year={2025},
  organization={Springer}
}
@inproceedings{du2024teach,
  title={Teach clip to develop a number sense for ordinal regression},
  author={Du, Yao and Zhai, Qiang and Dai, Weihang and Li, Xiaomeng},
  booktitle={European Conference on Computer Vision},
  pages={1--17},
  year={2024},
  organization={Springer}
}
```
