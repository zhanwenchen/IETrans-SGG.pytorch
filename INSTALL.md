## Installation

Most of the requirements of this projects are exactly the same as [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- PyTorch >= 1.2 (Mine 1.6.0 (CUDA 10.2))

- torchvision >= 0.4 (Mine 0.6.0 (CUDA 10.2))

- cocoapi

- yacs

- matplotlib

- GCC >= 4.9 (Mine 7.5.0)

- G++ (Mine 7.5.0)

- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name ietrans
conda activate ietrans

# this installs the right pip and dependencies for the fresh python
conda install python=3.11 ipython scipy h5py pandas

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.2
# conda install pytorch==1.6.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/waxnkw/IETrans-SGG.pytorch
cd IETrans-SGG.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR


```