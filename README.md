# Missile Defense Sim
This repository was created for the Purdue AAE 560 course on system of systems.

### Introduction
This repository simulates a missile defense scenario in which a nation must help defend against a volley of threat missiles. It uses the [OpenAI gym interface](https://github.com/openai/gym) where the **observation space** is an image of configurable size with the following channels all scaled to be from 0 to 255:

- [0] - n_missiles in threat platform at each pixel (0 for no threat platform)
- [1] - P_a of detectors within range at that pixel
- [2] - P_k of interceptors in range at that pixel

The **action space** is a pair of (latitude, longitude) points corresponding to where the next defense platform should be placed. The exact properties of this defense platform depend on if it is placed on land (ground based defense platform) or water (ship).

This environment is considered solved when all of the defended assets are still in tact at the end of the an episode.

### Getting Started

1. In an anaconda prompt, navigate to this folder and run "conda env create -f environment.yml".

2. Activate your new environment by running "conda activate mds_env"

3. Run the simulation using "python main.py"

Note you can see the command line options for main.py by running "python main.py --help"

4. To train an agent, run "python train.py"

This may be merged with main.py in the future.