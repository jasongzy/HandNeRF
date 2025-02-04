# HandNeRF & HandNeRF++

![teaser](https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/22978.png)

> [HandNeRF: Neural Radiance Fields for Animatable Interacting Hands](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_HandNeRF_Neural_Radiance_Fields_for_Animatable_Interacting_Hands_CVPR_2023_paper.pdf)
>
> Zhiyang Guo, Wengang Zhou, Min Wang, Li Li, Houqiang Li

## Installation

```shell
conda create -n handnerf python=3.10
conda activate handnerf

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
pip install --no-deps basicsr  # for RealESRGAN
```

Also, install [IntagHand](https://github.com/Dw1010/IntagHand):

```shell
cd tools
git clone https://github.com/Dw1010/IntagHand
pip install yacs fvcore
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

Then follow its [data preparation guide](https://github.com/Dw1010/IntagHand#pre-trained-model-and-data) to deploy the necessary data in `tools/IntagHand/misc`.

## Training

```shell
bash train_hands.sh
```

## Acknowledgement

- [zju3dv/animatable_nerf](https://github.com/zju3dv/animatable_nerf)
- [amundra15/livehand](https://github.com/amundra15/livehand)

## Citation

```bibtex
@inproceedings{guo2023handnerf,
  title={{HandNeRF}: Neural Radiance Fields for Animatable Interacting Hands},
  author={Guo, Zhiyang and Zhou, Wengang and Wang, Min and Li, Li and Li, Houqiang},
  booktitle={CVPR},
  year={2023}
}

@article{guo2024handnerfpp,
  title={{HandNeRF++}: Modeling Animatable Interacting Hands with Neural Radiance Fields},
  author={Guo, Zhiyang and Zhou, Wengang and Wang, Min and Li, Li and Li, Houqiang},
  year={2024}
}
```
