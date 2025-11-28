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

## 0. Argparser and AppLaunch()
- Define the simulator launcher function and which custom arguments it can receive to customize different ways to launch the simulation.

## 0. Import Libraries
- Import libraries for assets, scene, sim, utils(config class)

## 1. Scene Design and Configuration: CartpoleSceneCfg() Class
- Creates a scene configuration class
- Defines which entities (objects) that will be part of the scene and which configs they will hold - like casting actors for a scene.
- Configures these entities:
  - Ground
  - Light
  - Assets (Cartpole)

## 2. Run_simulator()
- Defines how the simulation will run (simulation loop). passing into it the entities defined by CartpoleSceneCfg()
- Like the scene rehearsal, scenario building, set construction etc. 

## 3. Main()
- Builds the scene according to the config class and stores it into a scene object
- Calls rum_simulator() passing the scene config object as a param - when the director calls "action", when all pieces as in place, and the scene actually runs. 

# tutorials\02_scene: create_scene.py

## 0. Argparser and AppLaunch()

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

## 0. Import Libraries

```py
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip
```

## 1. Scene Design and Configuration: CartpoleSceneCfg()

- Objects (entities) configuration
- Define Ground and Lights
  - 1. AssetBaseCfg(): Creates a Python config object
  - 2. Where to create?: prim_path="/World/defaultGroundPlane": Defines where in the folder tree to store (in-memory) the instance to be created, and passes it to the function's "path" parameter. 
  - 3. What to create?: spawn=sim_utils.GroundPlaneCfg(): Uses GrounPlaneCgf(), a predefined Python config class that comes with Isaac Lab and contains default parameters for generating a ground plane when spawned, and passes it to the function's "spawn" parameter.
- Any field you put in the config class that is an AssetBaseCfg, ArticulationCfg, RigidObjectCfg, etc. is automatically treated as a scene entity.

```py
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
```

- Set the Cartpole by using the CARTPOLE_CFG configuration object.
```py
    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

## 2. Run_simulator()

- Defines the simulation function 
- The simulator takes context and entities to be used in the scene (objects: robots, sensors, props etc) as params
  - This makes the run_simulator() function modular. You can later add more objects (entities) to it. The scene then automatically clones these entities across environments and updates them each simulation step.
- InteractiveScene is a manager that keeps track of all important simulation objects in your environment—robots, props, sensors, etc. 
```py
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
```

- Extracts scene entities, accesses the cartpole object and stores it in the "robot" variable
- Uses the InteractiveScene object "scene" (passed as a param of this function) to do “Give me the robot entity that was registered under the name cartpole.” during the scene entity definition above, CartpoleSceneCfg().
```py
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["cartpole"]
```

- Start simulation loop
- Resets it periodically and writes data to the simulation
```py
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
```

- Applies random forces to simulate someone moving the carpoles around
```py
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
```
- Steps the simulation and update buffers
```py
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)
```

## 3. Main()

- Apply configurations to the simulation 
```py
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
```
- set camera
- Instantiates a scene configuration class
- CartpoleSceneCfg() Class creates a scene recipe. It includes how many environments to make and how far apart they should be, plus the entities defined in the class definition (ground, light, cartpole). It's passed into the ``scene_cfg``, a config/data object describing the scene - like a json flat file. 
- InteractiveScene(scene_cfg) uses that recipe to actually build the scene in memory: it creates the USD prims, registers the robot, and sets up all env copies. Stores it into the ``scene``, an InteractiveScene object that contains the actual USD prims, entities, and environment clones - the actual constructed scene. 
```
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
```
- Reset and run the simulation
- Calls run_simulator() passing the camera and scene object as params 
```
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```
