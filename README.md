# isaaclab-RL-tutorials

- This readme provides a more structured and beginner-friendly walkthrough of official tutorials by Nvidia and other sources. 
- It follows the RL tutorials by LycheeAI: https://lycheeai-hub.com/isaac-lab/intermediate-videos and adds extra information to them. 
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
- Configures Ground, Lights and the Cartpolt
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
- Benefits: This allows for modularity and practicality when multiple developers are working on the same code. 


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

- Creates a Manager-Based Reinforcement Learning Environment ``ManagerBasedRLEnv()`` with the configurations previously defined ``CartpoleEnvCfg()``
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

# DIRECT WORKFLOW:

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
    # env
    decimation = 2 # Rendering steps per env step
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1 # Because here we only have one value for the action output - the force applied to the cartpole
    observation_space = 4 # Represent cartpole's and pole's position and velocity
    state_space = 0

    # Define simulation settings
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Define Robot's configuration (cartpole)
    # robot
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
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
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
```
    # Prepare actions before physics step - scale action force
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    # Set joint effort, specify joint index
    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    # Get observations and store them into a dictionary object
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

    # Compute Total Rewards
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

    # Defines when each environment should be terminated. Gets completion status of environments
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    # Resets environments and takes objects to their initial position (to random initial positions)
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

### 2.3: Define the function to compute rewards

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

