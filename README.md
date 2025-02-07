# TRAVERSE utils

## Environment Setup

Copied from the ['Getting Started'](https://github.com/vita-epfl/UniTraj?tab=readme-ov-file#-quick-start) section of UniTraj:

0. Create a new conda environment

```bash
conda create -n unitraj python=3.9
conda activate unitraj
```

1. Install ScenarioNet: https://scenarionet.readthedocs.io/en/latest/install.html

2. Install Unitraj:

```bash
git clone https://github.com/vita-epfl/UniTraj.git
cd unitraj
pip install -r requirements.txt
python setup.py develop
```

You can verify the installation of UniTraj via running the training script:

```bash
python train.py method=autobot
```

The model will be trained on several sample data.

