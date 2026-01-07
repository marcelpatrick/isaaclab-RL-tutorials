# isaaclab-RL-tutorials

- This readme provides a more structured and beginner-friendly walkthrough of official tutorials by Nvidia and other sources. 
- It follows the RL tutorials by LycheeAI: https://lycheeai-hub.com/isaac-lab/intermediate-videos and adds extra information to them. 
- It uses files from the IsaacLab **GitHub Repo**: https://github.com/isaac-sim/IsaacLab 
- It goes through the main Python files in this IsaacLab GitHub repo and explains what they do.
- It runs scripts from inside: ``C:\Users\[YOUR USER]\IsaacLab\scripts\tutorials\...``, or wherever you have installed the github project.

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

## 1. Scene Design and Configuration: SceneCfg() Class
- Creates a scene configuration class
- Defines which entities (objects) that will be part of the scene and which configs they will hold - like casting actors for a scene.
- Configures these entities:
  - Ground
  - Light
  - Objects (eg. Robots, Cartpole etc)

## 2. Run_simulator()
- Defines how the simulation will run (simulation loop). passing into it the entities defined by SceneCfg()
- Like the scene rehearsal, scenario building, set construction etc. 

## 3. Main()
- Builds the scene according to the config class and stores it into a scene object
- Calls run_simulator() passing the scene_cfg() object as a param - that's when the director calls "action", when all pieces are in place, and the scene actually runs. 

# tutorials\02_scene: create_scene.py
Video 1: https://youtu.be/Y-K1cAvnSFI?si=3pq8CU0TxsP-OB7r

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
- Create **Config Files** and save them in specified paths so they can later be used to spawn the primitives on the scenes (Ground, Lights, the Cartpole etc).
  - 1. AssetBaseCfg(): Creates a Python config object
  - 2. Where to create?: prim_path="/World/defaultGroundPlane": Defines where in the folder tree to store (in-memory) the instance to be created, and passes it to the function's "path" parameter. 
  - 3. What to create?: spawn=sim_utils.GroundPlaneCfg(): Uses GrounPlaneCgf(), a predefined Python config class that comes with Isaac Lab and contains default parameters for generating a ground plane when spawned, and passes it to the function's "spawn" parameter.
    - `sim_utils` is an object from library `isaaclab.sim` (imported above) used to create standard config objects `spawn` for commonly used prims (ground, lights, meshes, sensors etc).
    - When the simulation environment is initialized, the Isaac Lab framework reads these config objects and uses the spawner functions to actually create the prims in Omniverse.
- Any field you put in the config class that is a AssetBaseCfg, ArticulationCfg, RigidObjectCfg, etc. is automatically treated as a scene entity.

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
- "Create a variable ``cartpole`` of type ArticulationCfg (annotation), assign to it a configuration copied from the default config "
- Doing ``cartpole: ArticulationCfg =...`` is like doing ``my_int_number: int = 3``
- The annotation ``my_int_number: int = 3``prevents someone from assigning a value to that variable of the wrong type. - The IDE would show a warning while typing.  
```py
    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

## 2. Run_simulator()

- Defines the simulation function 
- The simulator takes context and entities to be used in the scene (objects: robots, sensors, props etc) as params
  - This makes the run_simulator() function modular. You can later add more objects (entities) to it. The scene then automatically clones these entities across environments and updates them each simulation step.
**InteractiveScene** is a manager that spawns primitives, keeps track of and manages all important simulation objects in your environment—robots, props, sensors, etc.
- This is what allows IsaacSim to render multiple robots side by side to run several training instances in parallel.
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
```py
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
```
- Reset and run the simulation
- Calls run_simulator() passing the camera and scene object as params 
```py
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

# MANUALLY RUN THE SIMULATION AND MDP FOR TESTING

- Here we will:
  1. Configure the Environment in the configuration file: `cartpole_env_cfg.py`
    - It already implements the MDP (Markov Decision Process) classes eg. `class ObservationsCfg`, `class RewardsCfg` but it doesn't yet learn from these rewards or observations. 
    - This simulation will be later registered in Gymnasium which will plug its observations and rewards to an RL algorithm which will enable learning. (explained down the line).
    - Nonetheless, this is code already being built to be compatible with the Gymnasium library 
  2. Execute the simulation manually with an execution script `run_cartpole_rl_env.py` to test if it's working properly
    - It manually steps the simulation with a while loop
 
# MANAGER WORKFLOW: 

- In the manager approach, standard functions (SceneCfg(), RewardsCfg(), ObservationsCfg(), Event() etc) are each defined inside their own separate classes, inside the ``cartpole_env_cfg.py`` file.
- Inside a different file ``run_cartpole_rl_env.py`` the class cartpole_env_cfg.py is imported and an object of this class is instantiated in main().
- Then, an env object is instantiated inheriting from ``ManagerBasedRLEnv()`` that takes the cfg object as param.
  - **ManagerBasedRLEnvironment** is a standard class provided by default as part of the Isaac Lab library.
  - It takes an environment configuration which contains all the defined aspects of the Markov Decision Process (MDP) such as actions, observations, rewards, and terminations. ``env = ManagerBasedRLEnv(cfg=env_cfg)``
- The env object is passed inside a loop to, in every step, take the input actions and return the outputs
  - In every step, it applies actions to the environment ``joint_efforts`` (**INPUTS**) and collects observations (new state after the joint_efforts (actions) have been applied), rewards, terminations etc (**OUTPUTS**) 
```py
obs, rew, terminated, truncated, info = env.step(joint_efforts)
```
- Benefits: 
  - Modularity: easy to remove the reward function or observation type and plug in a different to test how the model behaves. Plug and play different standard functions as standalone components
  - Encapsulation: safer, easier debugging, allows diff developers to work on the code at the same time. 


## CONFIGURATION SETUP: cartpole_env_cfg.py
- C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\cartpole_env_cfg.py


## 0.Import Libraries

```py
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
```

## 1. Scene Design and Configuration: CartpoleSceneCfg()

- Objects (entities) configuration
- Configures Ground, Lights and the Cartpolt
- **CARTPOLE_CFG**: the instance of the predefined cart-pole configuration object (``from isaaclab_assets.robots.cartpole import CARTPOLE_CFG``) that defines the robot's basic attributes (joints, links, limits, physics).

```py
##
# Scene definition
##


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    # Create a variable "robot" of type ArticulationCfg (annotation), assign to it a configuration copied from the default config template "CARTPOLE_CFG", and save it to this "path" whenever it gets rendered. 
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

## 2. Markov Decision Process (MDP) Settings / Reward Functions 

### 2.1. Action Configuration Class
- Defines **Action definitions** which assign policy’s output values to real robot commands into physical units.
- Specifies how the raw RL action output (a number from the policy) is converted into a physical force/effort applied to the chosen joint. eg: So the policy outputs, for example, 0.3, and the action definition turns that into: 0.3 × scale (100) = 30 units of joint effort applied to the slider_to_cart joint

```py
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # action definition: “The agent controls the cart by applying effort/force to the slider_to_cart joint of the robot,” scaled by 100 so the RL policy’s output maps to meaningful physical forces
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
```

- ### 2.2: Observations Configuration Class
- Inputs into the deep network (X)
- It defines what information the robot’s brain (the RL policy) gets each step.
- “Collect the robot’s joint positions and velocities, package them into one vector, don’t add noise, and feed that to the policy every step.” So the RL agent learns using only those two signals about the cart-pole’s state: joint position ``joint_pos_rel``, ``joint_vel_rel``
  
```py
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # observes the relative joint position of all joints
        # "func" method searches through joints IDs. 
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

### 2.3. Event Configuration Class
- It defines how the robot resets at the start of each episode.
- Each EventTerm randomizes the cart and pole joint positions and velocities within given ranges, ensuring varied starting states so the RL agent learns a more robust policy.

```py
@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
```

### 2.4. Reward Configuration Class
- It defines how the agent is rewarded or penalized.
- The code gives positive reward for staying alive, penalizes failure, penalizes the pole being off-upright, and adds small penalties for cart and pole motion.
- Together, these incentives teach the agent to balance the pole steadily.

```py
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # is_alive and is_terminated are predefined helper functions inside the isaaclab_tasks.manager_based.classic.cartpole.mdp module.
    # They are not generic Python or Isaac Lab functions; they are task-specific MDP utilities provided by the cartpole MDP implementation to detect success or failure conditions.

    # POSITIVE REINFORCEMENT: REWARD: weight=1.0
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # NEGATIVE REINFORCEMENT: REWARD: weight=-2.0
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # NEGATIVE REINFORCEMENT: REWARD: weight=-1.0
    # (3) Primary task: keep pole upright
    # Punishes whenever the pole has position deviations away from the upright position
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )

    # NEGATIVE REINFORCEMENT: REWARD: weight=-0.01
    # (4) Shaping tasks: lower cart velocity
    # Punishes if the robot speeds too much
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )

    # NEGATIVE REINFORCEMENT: REWARD: weight=-0.005
    # (5) Shaping tasks: lower pole angular velocity
    # Punishes whenever the pole acquires angular velocities which are too high
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

### 2.5. Termination Class Configuration
- It defines when an episode should end.
- One rule ends the episode after a time limit; the other ends it if the cart moves outside the allowed range.
- These termination conditions tell the RL system when to reset and start a new episode.
- **Episode**:
  - An Episode is a sequence of interactions between the agent and the environment. When the agent finishes its "mission" the key sequence of actions it was predefined to perform in order to learn.
  - After each Episode, the accumulated rewards are calculated and the result is used to train the algorithm -> back-propagation. 

<img width="1753" height="589" alt="image" src="https://github.com/user-attachments/assets/434ddbb8-6f89-4dd1-b176-87651410e2fa" />

## 3. Carpole Environment Configuration Class

- It bundles all components of the cart-pole RL environment into one configuration by calling all other configuration classes defined above: scene setup, observations, actions, events, rewards, and termination rules.
- It also sets global parameters like episode length, step rate, rendering interval, and viewer position.
- This final config tells Isaac Lab how to build and run the full RL environment.

```py
##
# Environment configuration
##


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

## RUNNING THE SCRIPT: run_cartpole_rl_env.py
- ``C:\Users\[YOUR USER]\isaaclab\scripts\tutorials\03_envs\run_cartpole_rl_env.py``

## 0. Argparser and AppLauncher
```py
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
```

- Imports the configuration class defined above in ``cartpole_env_cfg.py``
```py
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
```

## 1. Main
- Runs the simulation loop
- Creates a Manager-Based Reinforcement Learning Environment with the configurations previously defined by the ``class CartpoleEnvCfg(ManagerBasedRLEnvCfg):`` (imported here above) inside the previously configured file: ``C:\Users\myali\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\cartpole_env_cfg.py``
```py
def main():
    """Main function."""
    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
```
- Run the simulation loop
```py
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
```
- Apply random forces
```py
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
```
- Fetch training observations, rewards etc ``obs, rew, terminated, truncated, info`` in every step to be used as feedback into the model for learning
```py
            # step the environment
            # COLLECTS OUTPUT                     = TAKES INPUT
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```
## 2. RUN

in the project root directory `C:\Users\myali\isaaclab>` run `python scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-v0`

# DIRECT WORKFLOW:

- Defines all standard functions (observation, rewards, termination) inside the same script ``cartpole_env.py``

## cartpole_env.py
- Path: ``C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\direct\cartpole\cartpole_env.py``

## 0. Imports

```py
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
```

## CONFIGURATION CLASSES:
- Define the configuration classes for Cartpole

## 1. Cartpole Configuration Class: CartpoleEnvCfg()
- Defines the simulation environment configurations for: simulation environment, simulation mechanics, the robot, the scene, reset, and rewards
- Inherits from ``DirectRLEnvCfg`` configuration baseclass

```py
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):

# Define environment settings
    # How many physics steps to skip before the AI makes a new decision.
    # Physics runs at 120Hz, so decimation=2 means AI decides at 60Hz.
    # Saves compute while still allowing smooth simulation.
    decimation = 2
    # Max time (seconds) for one training episode before auto-reset.
    # Shorter = faster learning cycles; longer = tests long-term balance.
    # 5 seconds is enough to see if the pole stays upright.
    episode_length_s = 5.0
    # Multiplier for AI output to convert it into force (Newtons).
    # AI outputs small values (-1 to 1), this scales them to real forces.
    # 100N is strong enough to move the cart quickly.
    action_scale = 100.0  # [N]
    # Number of actions the AI can take. Here it's 1: push cart left or right.
    # Single value controls horizontal force on the cart.
    action_space = 1
    # Number of values the AI "sees" to make decisions: pole angle, pole speed,
    # cart position, cart speed. These 4 inputs describe the full system state.
    observation_space = 4
    # Extra state info for advanced training (e.g., asymmetric actor-critic).
    # 0 means we don't use additional privileged information here.
    state_space = 0

    # Define simulation settings
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Define Robot's configuration (cartpole)
    # This line configures the robot asset (cartpole) to be spawned in each parallel simulation environment.
    # uses CARTPOLE_CFG: A pre-defined articulation configuration object (imported from isaaclab_assets.robots.cartpole) that contains all the physical properties, USD file path, and joint/actuator settings for the cartpole robot
    # The path passed here defines where to save each robot instance
    # "ArticulationCfg" is a type annotation indicating that the robot_cfg variable is of type "Articulation object", a predefined IsaacLab class
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # Define how many environments the simulation will use and how they are spaced from each other
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # Define reset conditions. 
    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # Define reward parameters
    # reward scales
    # Bonus for each step the pole stays upright, encouraging survival.
    rew_scale_alive = 1.0
    # Penalty when episode ends due to failure, discouraging falling over.
    rew_scale_terminated = -2.0
    # Penalty for pole angle deviation from vertical, encouraging upright balance.
    rew_scale_pole_pos = -1.0
    # Small penalty for cart velocity, encouraging smooth/minimal cart movement.
    rew_scale_cart_vel = -0.01
    # Small penalty for pole angular velocity, encouraging stable non-swinging behavior.
    rew_scale_pole_vel = -0.005
```

## ENVIRONMENT DEFINITION

## 2. Setup the environment: CartpoleEnv()
- Inherits from ``DirectRLEnv``  

### 2.1: Define the Environment and Setup Scene

```py
class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    # Define cartpole environment and initialize it
    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    # Initializes cartpole articulation, sets the ground plane, clones, replicates multiple environments and "casts" the articulation to the scene. 
    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
```

### 2.2: Define Markov Decision Process settings
```py
# =============================================================================
    # _pre_physics_step: Prepares actions before the physics engine runs.
    # This function scales the raw AI output (typically -1 to 1) into real-world
    # force values (Newtons). It's called once per AI decision step before physics
    # simulation runs. Important because it bridges AI decisions to physical forces.
    # =============================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    # =============================================================================
    # _apply_action: Sends the prepared forces to the cart's motor/joint.
    # This function actually applies the computed force to move the cart left/right.
    # It's called every physics step to continuously apply the AI's chosen action.
    # Essential for translating AI decisions into actual cart movement in simulation.
    # Defines which joint to apply the force to - by joind_ids which are stored in tensors/arrays containign one type of joint in 
    # each index: [0] for cart or [1] for pole.
    # =============================================================================
    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    # =============================================================================
    # _get_observations: Gathers the current state info the AI needs to make decisions.
    # Returns 4 values: pole angle, pole angular velocity, cart position, cart velocity.
    # This is the AI's "eyes" - it can only see these 4 numbers to decide what to do.
    # Critical because good observations help the AI learn the relationship between
    # state and optimal actions for balancing the pole.
    # =============================================================================
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # =============================================================================
    # _get_rewards: Reward Function: Calculates how well the AI is doing at each step.
    # Combines multiple reward components: alive bonus, termination penalty,
    # pole angle penalty, cart velocity penalty, and pole velocity penalty.
    # This is the AI's "report card" - it learns by maximizing this reward signal.
    # The reward function shapes what behavior the AI learns (keep pole upright,
    # don't move cart too fast, stay smooth).
    # =============================================================================
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    # =============================================================================
    # _get_dones: Determines when an episode should end (success or failure).
    # Returns two signals: "terminated" (pole fell or cart went too far = failure)
    # and "time_out" (max episode time reached = neutral ending).
    # Important because it defines the boundaries of training episodes - the AI
    # learns that letting the pole fall past 90° or pushing the cart off-limits
    # ends the episode (bad), while surviving the full time is good.
    # =============================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    # =============================================================================
    # _reset_idx: Resets specific environments to their starting conditions.
    # When an episode ends (pole falls or time runs out), this function resets
    # that environment: puts cart back to center, gives pole a random small angle.
    # The random initial angle (from initial_pole_angle_range) ensures the AI
    # learns to balance from various starting positions, not just one pose.
    # Essential for continuous training - allows failed episodes to restart
    # immediately while other parallel environments keep running.
    # =============================================================================
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

#### 2.2.1: Define the function to compute rewards

- `compute_rewards()`: Core reward calculation function for the cartpole RL task.

- Purpose: Computes a scalar reward signal that guides the AI to learn pole balancing. This is the "feedback" the agent receives after each action - higher rewards mean better behavior, teaching the AI what actions lead to successful balancing.

- Takes current states from `_get_rewards()` to calculate reward. States is what defines whether the robot achieved its goal (generate positive reward) or not (generate negative reward). eg:

| Scenario | `pole_pos` | `pole_vel` | `cart_vel` | `terminated` | Total Reward |
|----------|------------|------------|------------|--------------|--------------|
| Perfect balance | 0.0 | 0.0 | 0.0 | False | **+1.0** |
| Slight tilt | 0.3 | 0.5 | 1.0 | False | **≈ +0.90** |
| Falling over | 1.2 | 3.0 | 4.0 | False | **≈ -0.50** |
| Episode ended | — | — | — | True | **-2.0** |

- Why Separate Functions?: Enables fast parallel computation across 4096+ environments

```py

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # POSITIVE REWARD: staying alive (environment not terminated)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # NEGATIVE REWARDS:
    # Negative reward for termination
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # Negative rewards to discount excessive movement
    # Penalises pole's angular deviation 
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # Penalises cart velocity deviation
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # Penalises pole velocity deviation 
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # CALCULATE TOTAL REWARD: sum of all previous rewards
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
```
## 3. RUN

On the VS Code terminal, input `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Cartpole-Direct-v0` and hit enter.

# TRAIN: IMPLEMENT REINFORCEMENT LEARNING

- Once we manually executed and tested the environments, now we will plug the observations and rewards from the previously implemented MDP into an RL algorithm for learning
- For this, we will use the OpenAI Gymnasium library

https://gymnasium.farama.org/index.html 
Video: https://www.youtube.com/watch?v=BSQEYj3Wm0Q&list=PLQQ577DOyRN_hY6OAoxBh8K5mKsgyJi-r&index=9

## Gymnasium

In the world of Reinforcement Learning (RL), you have two distinct sides:
1. The Environment: The world where the robot lives (Isaac Sim/Isaac Lab). This involves complex physics, friction, lighting, and USD stages.
2. The Agent: The AI brain (neural network) that wants to control the robot. It doesn't know anything about physics or 3D rendering; it only understands numbers (data inputs and action outputs).

**Gymnasium** is the standard protocol or library that sits in the middle so that any Agent can interact with Environments without needing to know how the physics engine works. It is a framework that wraps the code needed to run any environment and provides standard API functions that represent the main actions that simulations need to perform. 

- Registration: Once your code is registered within Gymnasium, it can be easily accessed from anywhere by using these templated API calls. It defines a standard "controller interface". eg: every environment must have a reset() function (start a new simulation) and a step() function (take an action, see what happens)
  - the step() function (standard API function from Gymnasium) calls the sim execution function from the original code, eg "CartpoleEnv()", which runs the sim. 
- It's a way to wrap your RL code inside a class. The class provides users the ability to start new episodes, take actions and visualize the agent’s current state through standard API functions. 
- It already comes with multiple standard environments for training robots with RL, but users can register their own envs.

Some of these main functions are: 
- Register: `gym.register(id, entry_point, **kwargs)` — add an environment name and how to create it so others can instantiate it by that name.
- Make: `gym.make(id, **kwargs)` — create an environment instance from a registered name with one call.
- Reset: `env.reset()` — start or restart an episode and return the initial observation (and info).
- Step: `env.step(action)` — apply an action, advance the sim, and return (observation, reward, terminated, truncated, info).
- Close: `env.close()` — release windows/processes/resources used by the environment.
- Spaces: `env.observation_space / env.action_space` — describe the shape, type and bounds of observations/actions so agents format data correctly.
- Render: `render()` shows or returns a visual frame of the environment so you can see what the simulator is doing (for debugging, recording, or human viewing).
- Wrappers: `gym.wrappers.* (e.g., RecordVideo, TimeLimit)` — add recording, time limits, or transforms. Allows users to modify or adapt its interface without changing the original code

## Register Gym Environments: __init__.py
- Environment Registry: C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\direct\cartpole\__init__.py
- It tells the Gymnasium interface which env config class to import: `entry_point=f"{__name__}.cartpole_env:CartpoleEnv"`

```py
# Env registration within Gymnasium
gym.register(
    # ID used to create and locate the env. The "new easier name" for this environment that can be later used to reference and instantiate it.
    id="Isaac-Cartpole-Direct-v0",
    # Fetches the path and name of the environment
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    # Key Woed arguments: allows you to pass specific configuration parameters to the environment
    kwargs={
        # References the python script that configures the env: cartpole_env_cfg: C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\cartpole\cartpole_env_cfg.py
        "env_cfg_entry_point": f"{__name__}.cartpole_env:CartpoleEnvCfg",
        # RL libraries: for this example we'll be using SKRL
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```

## Train Script: train.py
- The training script with the SKRL framework: C:\Users\[YOUR USER]\isaaclab\scripts\reinforcement_learning\skrl\train.py

This script is a **unified training entry point** for training reinforcement learning (RL) agents using the [skrl](https://skrl.readthedocs.io) library within the Isaac Lab framework. It's essentially a "launch script" that orchestrates the entire training pipeline for robot learning tasks.

- It uses:
**SKRL library**
  - Uses the PPO algorithm:
  - PPO: config file: C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\direct\cartpole\agents\skrl_ppo_cfg.yaml

**Hydra** is a configuration management framework that loads settings from YAML files and allows command-line overrides (like --num_envs 4). It keeps hyperparameters and environment settings separate from code, making experiments reproducible and easy to modify.

---

### What It Does  

1. **Parses Command-Line Arguments** — Accepts user inputs:

2. **Launches Isaac Sim** — Uses `AppLauncher` to boot up the NVIDIA Isaac Sim physics simulator.

3. **Configures the Environment** — Loads the environment configuration using Hydra, then overrides with CLI arguments.

4. **Sets Up Logging** — Creates timestamped directories under `logs/skrl/` to store configs, metrics, checkpoints, and videos.

5. **Creates the Training Environment** — Uses `gym.make()` to instantiate the environment, then wraps it for skrl compatibility.

6. **Runs Training** — Uses skrl's `Runner` class to execute the RL training loop.

7. **Cleans Up** — Closes the environment and simulation app when done.

### Flow: 
`train.py` → `import isaaclab_tasks` (triggers registration) → `gym.make("Isaac-Cartpole-Direct-v0")` (looks up registry) → `__init__.py` (contains `gym.register()`) → `cartpole_env.py` (`CartpoleEnv` + `CartpoleEnvCfg`) → trained policy

### Why Is It Useful?

Without this script, you would need to manually write boilerplate to launch Isaac Sim, instantiate environments, configure logging, set up multi-GPU training, handle checkpointing, and connect to your RL library—repeating this for every experiment. This script abstracts all of that into a single command: eg. `python train.py --task Isaac-Humanoid-Direct-v0 --num_envs 1024`

```py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

# =============================================================================
# SECTION 1: PARSE COMMAND-LINE ARGUMENTS
# =============================================================================


import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# =============================================================================
# SECTION 2: LAUNCH ISAAC SIM
# Uses AppLauncher to boot up the NVIDIA Isaac Sim physics simulator.
# =============================================================================

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
# This block determines which configuration file to load for the RL agent. eg "skrl_ppo_cfg.yaml"
# The "agent" is the AI brain that learns to control the robot through trial and error.
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# ===================================================================================
# MAIN TRAINING FUNCTION
# This is the core function that sets up and runs the training process.
# It takes configuration settings (loaded from YAML files via the decorator above)
# and command-line arguments to:
#   1. Configure the simulation environment (how many robots to simulate, which GPU to use)
#   2. Set up the AI agent that will learn to control the robot
#   3. Run the training loop where the agent learns through trial and error
# ===================================================================================
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""

    # =========================================================================
    # SECTION 3: CONFIGURE THE ENVIRONMENT
    # Loads the environment configuration using Hydra, then overrides with
    # CLI arguments (num_envs, device, seed, max_iterations, etc.).
    # =========================================================================

    # ====================These functions override default env configs with command line args===========
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # =========================================================================
    # SECTION 4: SET UP LOGGING
    # This section creates a folder structure to save all training data and results.
    # During training, the AI agent's progress (performance metrics, learned behaviors, 
    # configuration settings, and optional video recordings) are saved to these folders.
    # This makes it easy to track experiments, compare different training runs, and 
    # resume training later if needed.
    # =========================================================================

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # =========================================================================
    # SECTION 5: CREATE THE TRAINING ENVIRONMENT
    # Uses gym.make() to instantiate the environment, then wraps it for
    # skrl compatibility and optional video recording.
    # =========================================================================

    # Create the Gymnasium environment where the AI agent will learn.
    # The "task" defines what scenario to simulate (e.g., cartpole, robot arm).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # =========================================================================
    # SECTION 6: RUN TRAINING
    # Uses skrl's Runner class to execute the RL training loop.
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # =========================================================================

    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # =========================================================================
    # SECTION 7: CLEAN UP
    # Closes the environment and simulation app when done.
    # =========================================================================

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```

## Run Training
- run the training script
- Open a Terminal inside VS Code and run the command:
  - For Direct mode: `python scripts\reinforcement_learning\skrl\train.py --task Isaac-Humanoid-Direct-v0 --num_envs 4`
