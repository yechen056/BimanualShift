# INSTALLATION

#### 1. Environment

First, we create the rlbench environment and install the dependencies

```bash
conda create -n rlbench python=3.8
conda activate rlbench
git clone https://github.com/yechen056/BimanualShift.git
cd BimanualShift
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the [PyRep fork](https://github.com/markusgrotz/PyRep); reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd third_party
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

#### 3. RLBench

PerAct^2 uses the [RLBench fork](https://github.com/markusgrotz/RLBench). 

```bash
cd third_party
cd RLBench
pip install -e .
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

PerAct^2 uses the [YARR fork](https://github.com/markusgrotz/YARR).

```bash
cd third_party
cd YARR
pip install -e .
```

#### 5. pytorch3d

```bash
cd third_party
cd pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -e .
```

#### 6. Grounded-Segment-Anything

The current BimanualShift visual tracker uses the local Grounded-Segment-Anything checkout under `third_party/Grounded-Segment-Anything`.

```bash
cd /home/yechen/BimanualShift/third_party/Grounded-Segment-Anything
pip install -e .
```

Then build GroundingDINO in place:

```bash
cd /home/yechen/BimanualShift/third_party/Grounded-Segment-Anything/GroundingDINO
python setup.py build_ext --inplace
```

Make sure the following checkpoints exist:
- `/home/yechen/BimanualShift/third_party/Grounded-Segment-Anything/groundingdino_swint_ogc.pth`
- `/home/yechen/BimanualShift/third_party/Grounded-Segment-Anything/sam_vit_h_4b8939.pth`

You can download them with:
```bash
cd /home/yechen/BimanualShift/third_party/Grounded-Segment-Anything
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### 7. Local BERT Model

Download `bert-base-uncased` from:
- https://huggingface.co/google-bert/bert-base-uncased/tree/main

Place the model under `third_party/bert-base-uncased`:

```bash
cd /home/yechen/BimanualShift
mkdir -p third_party/bert-base-uncased
```

Once this directory exists, the project will load the tokenizer and model from the local path automatically.
