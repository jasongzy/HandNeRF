## Dataset

Download [InterHand2.6M_30fps](https://mks0601.github.io/InterHand2.6M/) and extract it into the `data` folder with the the following structure:

```
${PROJECT ROOT}
|-- data
|   |-- InterHand2.6M_30fps
|   |   |-- images
|   |   |   |-- train
|   |   |   |   |-- Capture0 ~ Capture26
|   |   |   |-- val
|   |   |   |   |-- Capture0
|   |   |   |-- test
|   |   |   |   |-- Capture0 ~ Capture7
|   |   |-- annotations
|   |   |   |-- skeleton.txt
|   |   |   |-- subject.txt
|   |   |   |-- train
|   |   |   |-- val
|   |   |   |-- test
```

## Dependencies

```shell
conda create -n interhand python=3.10
conda activate interhand

# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

sudo apt update
sudo apt install libboost-dev  # required by psbody-mesh

pip install -r requirements.txt
git clone https://github.com/vchoutas/smplx --depth=1
cd smplx && python setup.py install
cd .. && rm -rf smplx
```

In `render_mesh.py`, we need to render images using [pyrender](https://pyrender.readthedocs.io/en/latest/index.html).

> If you want to render scenes offscreen but don’t want to have to install a display manager or deal with the pains of trying to get OpenGL to work over SSH, you have two options.
>
> The first (and preferred) option is using EGL, which enables you to perform GPU-accelerated rendering on headless servers. However, you’ll need EGL 1.5 to get modern OpenGL contexts. This comes packaged with NVIDIA’s current drivers, but if you are having issues getting EGL to work with your hardware, you can try using OSMesa, a software-based offscreen renderer that is included with any Mesa install.

If you plan to use OSMesa (instead of EGL) as the backend, just switch to `os.environ["PYOPENGL_PLATFORM"] = "osmesa"` at the beginning of `render_mesh.py`. Note that to get OSMesa working, some extra steps have to be taken:

```shell
wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg --force-all -i ./mesa_18.3.3-0.deb
sudo apt install -f
pip install git+https://github.com/mmatl/pyopengl#egg=PyOpenGL
```

## Usage

To process a single sequence, run the python scripts in the specified order:

```shell
# extract camera annotations and MANO params
python get_annots.py --capture 0 --seq 0051_dinosaur
# produce blend_weights
python prepare_blend_weights.py --capture 0 --seq 0051_dinosaur
# produce masks and depth maps
python render_mesh.py --capture 0 --seq 0051_dinosaur
```

The script `process.sh` can be used for multi-sequence processing.

## Visualization

- `dataset_vis.ipynb`
- `new_anno_vis.ipynb`

## Other tools

- `merge_seq.py`: merge specified sequences of InterHand2.6M into a new one, including images and all necessary annotations. No extra sapce is occupied since symbolic links are used. Example: `python merge_seq.py --seq 0006_thumbup_relaxed 0012_aokay_upright 0041_claws 0045_shakespearesyorick --new_name various_poses`.
- `gif.py`: generate GIF demos for all sequences in a specified Capture of InterHand2.6M. Example: `python gif.py --capture 0`.

## Reference

- [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html#getting-pyrender-working-with-osmesa)
