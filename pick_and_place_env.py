import pybullet_data
import os
from moviepy.editor import ImageSequenceClip

from constants import *
from robot_suction import SuctionGripper

from scripts.run_pipeline_single_obj_parallel import main_run_pipeline_single_obj_parallel, store_assembly
from bloxnet.pipelines.whole_structure import get_outcome_reason, replan_assembly
from utils import *

# Custom pick-and-place environment class using PyBullet
class PickPlaceEnv():
  def __init__(self, gui=True,json_file="blocksets/some_cuboid_blocks.json",max_seq_iterations=10):
    # Simulation config
    self.dt = 1/480
    self.sim_step = 0

    # Start PyBullet with or without GUI
    if gui:
      pybullet.connect(pybullet.GUI)  
      pybullet.setRealTimeSimulation(1)
    else:
      pybullet.connect(pybullet.DIRECT)

    # Physics and search paths
    pybullet.setPhysicsEngineParameter(enableFileCaching=0)
    assets_path = os.path.dirname(os.path.abspath(""))
    pybullet.setAdditionalSearchPath(assets_path)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setTimeStep(self.dt)

    # UR5e configuration
    self.home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  # Joint angles: (J0, J1, J2, J3, J4, J5).
    self.home_ee_euler = (np.pi, 0, np.pi)  # (RX, RY, RZ) base rotation in Euler angles.
    self.ee_link_id = 9  # Link ID of UR5 end effector.
    self.tip_link_id = 10  # Link ID of gripper finger tips.
    self.gripper = None

    # Blockset, file paths, number of iterations
    self.blockset_pos = {}
    self.blockset_json_file = json_file
    self.available_blocks = []
    self.max_seq_iterations = max_seq_iterations
    
  # IDfRA main loop — refine over multiple iterations
  def run_idfra_pipeline(self):
    user_input = input("Enter structure name: ")
    plans_so_far = {}
    structure_dir_name = (user_input.lower()).replace(' ', '-')
    all_available_dims = get_unique_dim_triplets(block_names=self.available_blocks)   # For dimension switching

    for seq_iter in range(self.max_seq_iterations):
      if seq_iter != 0:
        #  Call Replanner to get revised assembly
        assembly = replan_assembly(to_build=user_input,json_file=self.blockset_json_file,outcome_reason=missing_suggestions,plans_so_far=str(plans_so_far),seq_iter=seq_iter)
      else:
        # Generate initial assembly using simplified BloxNet
        assemblies = main_run_pipeline_single_obj_parallel(user_input.lower(),num_structures=1,max_workers=4,json_file=self.blockset_json_file)

      missing_blocks = []
      if seq_iter != 0:
        #  Log and read action plan for new assembly
        store_assembly(assembly=assembly, seq_iter=seq_iter)  
        structure_list = load_from_json(f"./gpt_caching/{structure_dir_name}/replan_assembly_{seq_iter}/position_plan.json")
      else:
        # Initialise cumulative plan history
        with open(f"gpt_caching/{structure_dir_name}/responses/main_plan_{0}_response.md", "r", encoding="utf-8") as f:
          initial_plan = f.read()
        plans_so_far[f"iteration{seq_iter}"] = initial_plan     

        #  Log and read action plan for initial assembly
        store_assembly(assembly=assemblies[0], seq_iter=seq_iter)
        structure_list = load_from_json(f"./gpt_caching/{structure_dir_name}/initial_assembly_{0}/structure.json")
      print("------------- Plan generated, saved, and structure loaded -------------")

      blocks_moved = {}
      _ = self.reset()

      pick_place_actions = {} # Format: pick_obj_name : {'place' : place_target_xyz, 'rotation': rotation_ori}
      for action_dict in structure_list:
        if action_dict["shape"] == 'cuboid':
          pick_obj_name = f"{action_dict['dimensions']['x']}x{action_dict['dimensions']['y']}x{action_dict['dimensions']['z']}-{action_dict['shape']}"
        elif action_dict["shape"] == 'cylinder':
          pick_obj_name = f"{action_dict['dimensions']['radius']*2}x{action_dict['dimensions']['radius']*2}x{action_dict['dimensions']['height']}-{action_dict['shape']}"

        # Resolve dimension switch : map new dimensions to roll, pitch, yaw
        roll, pitch, yaw = 0, 0, 0
        if pick_obj_name not in self.available_blocks:
          these_dims = [action_dict['dimensions']['x'],action_dict['dimensions']['y'], action_dict['dimensions']['z']]
          for avail_dims in all_available_dims:
            if sorted(these_dims) == sorted(avail_dims):
              print(these_dims, " === ", avail_dims)
              pick_obj_name = f"{avail_dims[0]}x{avail_dims[1]}x{avail_dims[2]}-{action_dict['shape']}"
              swap = get_axis_swap(avail_dims, these_dims)
              if swap == (0, 2):
                print("X and Z swapped - pitch 90")
                roll, pitch, yaw = 0, 90, 0
              elif swap == (1, 2):
                print("Y and Z swapped - roll 90")
                roll, pitch, yaw = 90, 0, 0
              elif swap == (0, 1):
                print("X and Y swapped - yaw 90")
                roll, pitch, yaw = 0, 0, 90
              break

        # Configure object names to match dedicated data structure for configuration tracking
        if pick_obj_name in blocks_moved.keys():
          blocks_moved[pick_obj_name] += 1
          pick_obj_name = f"{pick_obj_name}_{blocks_moved[pick_obj_name]}"
        else:
          blocks_moved[pick_obj_name] = 1
          pick_obj_name = f"{pick_obj_name}_1"

        # Check for missing block (and skip to next action)
        if pick_obj_name not in self.blockset_pos.keys():
            missing_blocks.append(pick_obj_name)
            continue

        # Build Python dictrionary for robot execution
        pick_place_actions[pick_obj_name] = {}
        if seq_iter != 0:
          pick_place_actions[pick_obj_name]["place_loc"] = np.array([action_dict["position"]['x']/1000, (action_dict["position"]['y']/1000) - 0.5, action_dict["position"]['z']/1000 ])
          pick_place_actions[pick_obj_name]["rotation_ori"] = [roll, pitch, yaw+action_dict['yaw']] 
        else:
          pick_place_actions[pick_obj_name]["place_loc"] = np.array([action_dict["position"][0]/1000, (action_dict["position"][1]/1000) - 0.5, action_dict["position"][2]/1000 ])
          pick_place_actions[pick_obj_name]["rotation_ori"] = [roll, pitch, yaw+Rotation.from_quat(action_dict['orientation']).as_euler('xyz', degrees=True)[2]]    

      # Make all extra blocks transparent to remove clutter
      for obj_name in self.blockset_pos.keys():
        if obj_name not in pick_place_actions.keys():
          pybullet.changeVisualShape(self.obj_name_to_id[obj_name], linkIndex=-1, rgbaColor=[0, 0, 0, 0])

      for pick_obj_name in pick_place_actions.keys():
        if 'cylinder' in pick_obj_name: height_dim_idx = 1
        else: height_dim_idx = 2

        # Extract pick and place locations for action
        pick_target_xyz = np.array([self.blockset_pos[pick_obj_name]["position"][0], self.blockset_pos[pick_obj_name]["position"][1], self.blockset_pos[pick_obj_name]["position"][2] + self.blockset_pos[pick_obj_name]["dimensions"][height_dim_idx]/2 + 0.01])

        place_target_xyz = pick_place_actions[pick_obj_name]["place_loc"]
        place_target_xyz[2] = place_target_xyz[2] + self.blockset_pos[pick_obj_name]["dimensions"][height_dim_idx]/2
        rotation_ori = pick_place_actions[pick_obj_name]["rotation_ori"]

        # Resolve roll and pitch first on the side of the table
        if rotation_ori[0] != 0 or rotation_ori[1] != 0:
          if rotation_ori[0] != 0:
            _ = self.rotate_roll_pitch(pick_obj_name, roll_pitch=[rotation_ori[0], 0])
            height_dim_idx = 1
          if rotation_ori[1] != 0:
            _ = self.rotate_roll_pitch(pick_obj_name, roll_pitch=[0, rotation_ori[1]])
            height_dim_idx = 0
          block_id = self.obj_name_to_id[pick_obj_name]
          block_top_normal = get_top_normal(block_id)
          pick_orientation = quat_from_top_normal(block_top_normal)

          # Integrate yaw to compute place orientation of rolled/pitched block
          yaw_angle = np.deg2rad(rotation_ori[2])
          yaw_rot = Rotation.from_euler('z', yaw_angle).as_quat()
          place_ori = (Rotation.from_quat(yaw_rot) * Rotation.from_quat(pick_orientation)).as_quat()

          # Recompute pick target -- may be updated if roll or pitch used
          pick_target_xyz = np.array([self.blockset_pos[pick_obj_name]["position"][0], self.blockset_pos[pick_obj_name]["position"][1],self.blockset_pos[pick_obj_name]["position"][2] + self.blockset_pos[pick_obj_name]["dimensions"][height_dim_idx] / 2 + 0.01])
        else:
          # If there is no dimension switch
          pick_orientation = self.object_name_to_ori[pick_obj_name]

          # Place block with target yaw
          rx, ry, rz = self.home_ee_euler
          tilt_yaw = np.deg2rad(rotation_ori[2])
          place_ori = pybullet.getQuaternionFromEuler([rx, ry, rz + tilt_yaw])

        # Execute pick and place robot action
        action_exec = {"pick": pick_target_xyz, "place": place_target_xyz}
        self.step(action=action_exec, pick_orientation=pick_orientation, place_ori=place_ori)  

      # Log rendered assembly
      if seq_iter != 0: save_pybullet_img_position_plan(json_output=structure_list, structure_dir_name=structure_dir_name, seq_iter=seq_iter)

      # Create GIF for Judge and records
      debug_clip = ImageSequenceClip(self.cache_video, fps=10)
      base_file_path = f"./gpt_caching/{structure_dir_name}/{user_input.lower()}_video_{seq_iter}"
      debug_clip.write_gif(f"{base_file_path}_d.gif", fps=8)

      # Update cumulative plan history
      if seq_iter !=0:
        with open(f"gpt_caching/{structure_dir_name}/responses/re_plan_{seq_iter}_response.md", "r",
                  encoding="utf-8") as f:
          re_plan = f.read()
        plans_so_far[f"iteration{seq_iter}"] = re_plan

      for action_dict in structure_list:
        if seq_iter != 0:
          del action_dict["name"]
          del action_dict["color"]
        else:
          del action_dict["gpt_name"]
          del action_dict["block_name"]
          del action_dict["color"]

      print(">>>>>>> Blocks missing: ", missing_blocks)

      # Call Judge module for feedback
      missing_suggestions = get_outcome_reason(user_input.lower(), gif_path=[f"{base_file_path}_d.gif",f"{base_file_path}_l.gif"],json_file=self.blockset_json_file,seq_iter=seq_iter,structure_list=structure_list,missing_blocks=missing_blocks)

      self.cache_video = []

  # Reset simulation and configure environment.
  def reset(self):
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    pybullet.setGravity(0, 0, -9.8)
    self.cache_video = []

    # Temporarily disable rendering to load URDFs faster.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    # Add robot.
    pybullet.loadURDF("plane.urdf", [0, 0, -0.001])
    self.robot_id = pybullet.loadURDF("C:\\Users\\nishk\\OneDrive\\Desktop\\DT\\say_can\\ur5e\\ur5e\\ur5e.urdf", [0, 0, 0], flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    self.ghost_id = pybullet.loadURDF("C:\\Users\\nishk\\OneDrive\\Desktop\\DT\\say_can\\ur5e\\ur5e\\ur5e.urdf", [0, 0, -10])  # For forward kinematics.
    self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in range(pybullet.getNumJoints(self.robot_id))]
    self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

    # Move robot to home configuration.
    for i in range(len(self.joint_ids)):
      pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])

    # Add gripper.
    if self.gripper is not None:
      while self.gripper.constraints_thread.is_alive():
        self.constraints_thread_active = False
    self.gripper = SuctionGripper(self.robot_id, self.tip_link_id)
    self.gripper.release()

    # Add workspace.
    plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
    plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.3, 0.3, 0.001])
    plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
    pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])

    # Load objects according to JSON.
    self.obj_name_to_id = {}
    self.object_name_to_ori = {}

    blockset = load_from_json(self.blockset_json_file)

    for obj in blockset.keys():
      number_avail = blockset[obj]["number_available"]
      start_idx = obj.find('-') + 1 
      self.available_blocks.append(f"{obj[start_idx:]}")
      for i in range(1, number_avail+1):
        if 'cuboid' in obj:
          self.blockset_pos[f"{obj[start_idx:]}_{i}"] = {"dimensions" : [blockset[obj]["dimensions"]["x"]/1000, blockset[obj]["dimensions"]["y"]/1000, blockset[obj]["dimensions"]["z"]/1000], "position" : []}
        elif 'cylinder' in obj:
          self.blockset_pos[f"{obj[start_idx:]}_{i}"] = {"dimensions": [blockset[obj]["dimensions"]["radius"] / 1000, blockset[obj]["dimensions"]["height"] / 1000], "position": []}

    obj_xyz = np.zeros((0, 3))
    for obj_name in self.blockset_pos.keys():
      # Sample random, collision-free position.
      while True:
        xmin, xmax = BOUNDS[0,0] + 0.1, BOUNDS[0,1] - 0.1
        ymin, ymax = BOUNDS[1,0] + 0.1, BOUNDS[1,1] - 0.1
        rand_x, rand_y = sample_out_center_rect(xmin, xmax, ymin, ymax, excl_frac=0.55)

        rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
        if len(obj_xyz) == 0:
          obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
          break
        else:
          nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
          if nn_dist > 0.1:              # originally 0.15
            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
            break

      object_position = rand_xyz.squeeze()
      self.blockset_pos[obj_name]["position"] = object_position

      # Create and place the object.
      if 'cuboid' in obj_name:
        half_extents = [dim / 2 for dim in self.blockset_pos[obj_name]["dimensions"]]
        tilt_yaw = np.deg2rad(np.random.uniform(-180, 180))
        rx, ry, rz = self.home_ee_euler
        tilt_quat = pybullet.getQuaternionFromEuler([rx, ry, rz+tilt_yaw])
        object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half_extents)
        object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=half_extents)
        density = 1000  # Density of the block in kg/m^3
        mass = density * self.blockset_pos[obj_name]["dimensions"][0] * self.blockset_pos[obj_name]["dimensions"][1] * self.blockset_pos[obj_name]["dimensions"][2]
        object_id = pybullet.createMultiBody(mass, object_shape, object_visual, basePosition=object_position,baseOrientation=tilt_quat)

      elif 'cylinder' in obj_name:
        tilt_quat = pybullet.getQuaternionFromEuler(self.home_ee_euler)
        radius_m, height_m = self.blockset_pos[obj_name]["dimensions"][0], self.blockset_pos[obj_name]["dimensions"][1]
        object_shape = pybullet.createCollisionShape(pybullet.GEOM_CYLINDER, radius=radius_m, height=height_m)
        object_visual = pybullet.createVisualShape(pybullet.GEOM_CYLINDER, radius=radius_m, length=height_m)
        density = 1000  # Density of the block in kg/m^3
        mass = density * np.pi * radius_m ** 2 * height_m
        object_id = pybullet.createMultiBody(baseMass=mass, baseCollisionShapeIndex=object_shape, baseVisualShapeIndex=object_visual, basePosition=object_position,baseOrientation=tilt_quat)

      pybullet.changeDynamics(object_id, -1,
                              lateralFriction=0.5,
                              spinningFriction=0.2)
      pybullet.changeVisualShape(object_id, -1, rgbaColor=COLORS["green"])
      self.obj_name_to_id[obj_name] = object_id
      self.object_name_to_ori[obj_name] = pybullet.getBasePositionAndOrientation(object_id)[1]

    # Re-enable rendering and step simulation.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    for _ in range(200):
      pybullet.stepSimulation()
    return self.get_observation()
  
  # Move to target joint positions with position control.
  def servoj(self, joints):
    pybullet.setJointMotorControlArray(
      bodyIndex=self.robot_id,
      jointIndices=self.joint_ids,
      controlMode=pybullet.POSITION_CONTROL,
      targetPositions=joints,
      positionGains=[0.01]*6)
  
  # Move to target end effector position.
  def movep(self, position,ori=None):   
    if ori is None:
      ori = pybullet.getQuaternionFromEuler(self.home_ee_euler)
    joints = pybullet.calculateInverseKinematics(
        bodyUniqueId=self.robot_id,
        endEffectorLinkIndex=self.tip_link_id,
        targetPosition=position,
        targetOrientation=ori,
        maxNumIterations=100)
    self.servoj(joints)

  # Do pick and place motion primitive.
  def step(self, action=None,pick_orientation=np.array([]), place_ori=np.array([])):
    pick_xyz, place_xyz = action["pick"].copy(), action["place"].copy()

    # Set fixed primitive z-heights.
    hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.3])

    # Move to object.
    ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      self.movep(hover_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
      self.movep(pick_xyz, pick_orientation)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Pick up object.
    self.gripper.activate()
    for _ in range(240):
      self.step_sim_and_render()
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      self.movep(hover_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    
    # Move to place location.
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz, place_ori)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Place down object.
    while (not self.gripper.detect_contact()) and (place_xyz[2] > action["place"][2]+0.001):
      place_xyz[2] -= 0.001
      self.movep(place_xyz,place_ori)
      for _ in range(6):
        self.step_sim_and_render()
    self.gripper.release()
    for _ in range(240):
      self.step_sim_and_render()
    place_xyz[2] = 0.2
    ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    end_xyz = np.float32([0, -0.5, 0.2])
    while np.linalg.norm(end_xyz - ee_xyz) > 0.01:
      self.movep(end_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    return 

  # Rotate object by roll or pitch before placing.
  def rotate_roll_pitch(self, pick_obj_name="", roll_pitch=(0, 0)):
    # Compute new block height
    if roll_pitch[0] != 0:
      new_z = self.blockset_pos[pick_obj_name]["dimensions"][1]   # original y - because y and z swap
    elif roll_pitch[1] != 0:
      new_z = self.blockset_pos[pick_obj_name]["dimensions"][0]   # original x - because x and z swap

    action = {"pick": np.array( [self.blockset_pos[pick_obj_name]["position"][0], self.blockset_pos[pick_obj_name]["position"][1],
       self.blockset_pos[pick_obj_name]["position"][2] + self.blockset_pos[pick_obj_name]["dimensions"][2] / 2 + 0.01]),
              "place": np.array([-0.35, -0.4, new_z + 0.05])}

    pick_xyz, place_xyz = action["pick"].copy(), action["place"].copy()

    hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.3])

    # Move to object.
    ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      self.movep(hover_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    pick_orientation = self.object_name_to_ori[pick_obj_name]
    while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
      self.movep(pick_xyz, pick_orientation)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Pick up object.
    self.gripper.activate()
    for _ in range(240):
      self.step_sim_and_render()
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      self.movep(hover_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Move to place location with roll/pitch.
    tilt_roll = np.deg2rad(roll_pitch[0])     
    tilt_pitch = np.deg2rad(roll_pitch[1])   
    rx, ry, rz = self.home_ee_euler
    place_ori = pybullet.getQuaternionFromEuler([rx+tilt_roll, ry+tilt_pitch, rz])
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz, place_ori)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Place down object.
    while (not self.gripper.detect_contact()) and (place_xyz[2] > action["place"][2] + 0.001):
      place_xyz[2] -= 0.001
      self.movep(place_xyz, place_ori)
      for _ in range(3):
        self.step_sim_and_render()
    self.gripper.release()
    for _ in range(240):
      self.step_sim_and_render()
    place_xyz[2] = 0.2
    ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
    end_xyz = np.float32([0, -0.5, 0.2])
    while np.linalg.norm(end_xyz - ee_xyz) > 0.01:
      self.movep(end_xyz)
      self.step_sim_and_render()
      ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

    # Update internal state with new orientation.s
    new_position, new_orientation = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[pick_obj_name])
    self.blockset_pos[pick_obj_name]["position"] = [new_position[0], new_position[1], new_position[2]]
    self.object_name_to_ori[pick_obj_name] = new_orientation

    return new_orientation
          
  # Set alpha value for visual shape of robot links.
  def set_alpha_transparency(self, alpha: float) -> None:
    for id in range(20):
      visual_shape_data = pybullet.getVisualShapeData(id)
      for i in range(len(visual_shape_data)):
        object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
        rgba_color = list(rgba_color[0:3]) +  [alpha]
        pybullet.changeVisualShape(
            self.robot_id, linkIndex=i, rgbaColor=rgba_color)

  # Step sim and render video every 60 steps.
  def step_sim_and_render(self):
    pybullet.stepSimulation()
    self.sim_step += 1
    if self.sim_step % 60 == 0:
      self.cache_video.append(self.get_camera_image_diagonal())

  # Standard camera.
  def get_camera_image(self):
    image_size = (240, 240)
    intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
    color, _, _, _, _ = self.render_image(image_size, intrinsics)
    return color
  
  # Diagonal view.
  def get_camera_image_diagonal(self):
    image_size = (240, 240)
    intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
    color, _, _, _, _ = self.render_image(image_size,
                                          intrinsics,
                                          position=(+.25, -.8, 0.2),
                                          orientation=(np.pi/3, np.pi, (5*np.pi)/4 - np.pi/12),
                                        )
    return color

  # Top-down image and depth.
  def get_camera_image_top(self, 
                           # image_size=(240, 240),
                           image_size=(240, 240),
                           intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                           position=(0, -0.5, 5),
                           orientation=(0, np.pi, -np.pi / 2),
                           zrange=(0.01, 1.),
                           set_alpha=True):
    set_alpha and self.set_alpha_transparency(0)
    color, depth, _, _, _ = self.render_image_top(image_size,
                                             intrinsics,
                                             position,
                                             orientation,
                                             zrange)
    set_alpha and self.set_alpha_transparency(1)
    return color, depth

  # RGBD -> heightmap and xyzmap.
  def get_observation(self):
    observation = {}

    # Render current image.
    color, depth, position, orientation, intrinsics = self.render_image()

    # Get heightmaps and colormaps.
    points = self.get_pointcloud(depth, intrinsics)
    position = np.float32(position).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.float32(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    points = self.transform_pointcloud(points, transform)
    heightmap, colormap, xyzmap = self.get_heightmap(points, color, BOUNDS, PIXEL_SIZE)

    observation["image"] = colormap
    observation["xyzmap"] = xyzmap
    # observation["pick"] = list(self.config["pick"])
    # observation["place"] = list(self.config["place"])
    return observation

  def render_image(self,
                   image_size=(720, 720), 
                   intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1),
                   position=(0, -0.85, 0.4),
                   orientation=(np.pi / 4 + np.pi / 48, np.pi, np.pi)):
  
    # Camera parameters.
    position = position
    orientation = orientation
    orientation = pybullet.getQuaternionFromEuler(orientation)
    zrange = (0.01, 10.)
    noise=True

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
      color = np.int32(color)
      color += np.int32(np.random.normal(0, 3, color.shape))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
      depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics

  def render_image_top(self, 
                       image_size=(240, 240), 
                       intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                       position=(0, -0.5, 5),
                       orientation=(0, np.pi, -np.pi / 2),
                       zrange=(0.01, 1.)):

    # Camera parameters.
    orientation = pybullet.getQuaternionFromEuler(orientation)
    noise=True

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = (0.01, 10.)
    viewm = pybullet.computeViewMatrix(position, lookat, updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi
    aspect_ratio = image_size[1] / image_size[0]
    projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pybullet.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        # flags=pybullet.ER_NO_SEGMENTATION_MASK,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if noise:
      color = np.int32(color)
      color += np.int32(np.random.normal(0, 3, color.shape))
      color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.float32(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
    depth = (2 * znear * zfar) / depth
    if noise:
      depth += np.random.normal(0, 0.003, depth.shape)

    intrinsics = np.float32(intrinsics).reshape(3, 3)
    return color, depth, position, orientation, intrinsics

  # Convert depth image to 3D pointcloud.
  def get_pointcloud(self, depth, intrinsics):
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

  # Apply 4x4 transformation matrix to pointcloud.
  def transform_pointcloud(self, points, transform):
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            "constant", constant_values=1)
    for i in range(3):
      points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

  # Project 3D pointcloud into top-down heightmap.
  def get_heightmap(self, points, colors, bounds, pixel_size):
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
    xyzmap = np.zeros((height, width, 3), dtype=np.float32)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
      colormap[py, px, c] = colors[:, c]
      xyzmap[py, px, c] = points[:, c]
    colormap = colormap[::-1, :, :]  # Flip up-down.
    xv, yv = np.meshgrid(np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
                         np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width))
    xyzmap[:, :, 0] = xv
    xyzmap[:, :, 1] = yv
    xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
    heightmap = heightmap[::-1, :]  # Flip up-down.
    return heightmap, colormap, xyzmap