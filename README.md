## Requirements
Python 3.9+


### Required Python Packages
numpy
torch
gymnasium[mujoco]
mujoco
stable-baselines3
snntorch
matplotlib
tensorboard
tqdm

### install dependencies:
pip install -r requirements.txt

The code is divided into 3 files:

- Reinforcement.py
  This produces trajectory data used to train the diffusion model.
- Diffusion.py
  The model learns to generate future pose sequences conditioned on morphology and past observations.
- morphSNN.py
  Train PPO using the ALIF-based SNN feature extractor and diffusion-generated motion priors:

Code for training both the RL and Diffusion models is setup in the Train.ipynb notebook

After training, Generate.ipynb has utilities for analyzing the models
