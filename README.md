# isaaclab-RL-tutorials

- This readme follows the RL tutorials by LycheeAI: https://lycheeai-hub.com/isaac-lab/intermediate-videos and adds extra information to them. 
- It uses files from the IsaacLab **GitHub Repo**: https://github.com/isaac-sim/IsaacLab 
- It goes through the main Python files in this IsaacLab GitHub repo and explains what they do.
- It runs scripts from inside: ``C:\Users\[YOUR USER]\IsaacLab\scripts\tutorials\02_scene``, or wherever you have installed the github project.

# Setup

- Open the Anaconda Prompt terminal and activate the conda env you have created with isaacsim and isaaclab installed and all the dependencies and the isaacsim/isaaclab base github project: ``conda activate env_isaacsim``
- If you haven't done it yet you can follow this tutorial: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial
-Navigate to the folder containing all the scripts you need to run and type ``code .`` to open VS Code from inside your anaconda env. OR, in this case, after activating the environment, just type: ``code Isaaclab``
  - This will open VS Code with the correct python interpreter from this env and the VS code terminal will also run inside this env. 
- On the folder structure on the left, navigate to the isaaclab project or tutorial you want to run
- click the "run python file" button on VS code to run the script.

# Standard Functions
- Standard functions that appear in most IsaacLab code scripts

## 0. Argparser and AllLaunch()
- Define the simulator launcher function and which custom arguments it can receive to customize different ways to launch the simulation.

## 0. Import Libraries
- Import libraries for assets, scene, sim, utils(config class)

## 1. Scene Design and Configuration: CartpoleSceneCfg()
- Configures:
  - Ground
  - Light
  - Assets (Cartpole)

## 2. Run_simulator()

## 3. Main()


# tutorials\02_scene: create_scene.py

## 0. Argparser and AllLaunch()

```py
"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```



