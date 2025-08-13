import numpy as np
from bloxnet.utils.transform_utils import quat2mat, angular_error
from bloxnet.structure import Block
import pybullet as p
import pybullet_data
from bloxnet.pybullet.place_blocks_in_json import (
    move_until_contact,
    move_out_of_contact,
)
import time
from typing import List


class Structure:
    def __init__(self, available_blocks={}):
        self.structure: List[Block] = []
        self.available_blocks = available_blocks
        self.drop = True
        self.stability_physics = True
        self.sort_by_height = False

    def get_json(self):
        return [block.get_json() for block in self.structure]

    def get_gpt_json(self):
        return {
            block.gpt_name: {
                "block_type": block.block_name,
                "position": block.get_round_position(),
                "orientation": block.euler_orientation,
            }
            for block in self.structure
        }

    def _get_block_index_by_id(self, id):
        for i, block in enumerate(self.structure):
            if block.id == id:
                return i
        return None

    def get_block_by_id(self, id):
        idx = self._get_block_index_by_id(id)
        if idx is None:
            return -1
        return self.structure[idx]

    def set_block_by_id(self, id, block):
        idx = self._get_block_index_by_id(id)
        if idx is None:
            return -1
        self.structure[idx] = block

        self.place_blocks(drop=self.drop)
        return 0

    def add_block(self, block):
        self.structure.append(block)

    def add_blocks(self, blocks):
        self.structure.extend(blocks)

    def delete_by_id(self, id):
        idx = self._get_block_index_by_id(id)
        self.structure.pop(idx)

    def _place_block(self, block, drop=True, debug=False):
        id = create_block(block)

        if drop:
            # Move the block until it makes contact with another object
            move_until_contact(id, direction=[0, 0, -1], step_size=0.001, debug=debug)
            move_out_of_contact(id)

            position, orientation = p.getBasePositionAndOrientation(id)

            block.position = [pos * 1000 for pos in position]
            block.orientation = orientation
            block.id = id

    def place_blocks(self, drop=True, debug=False):
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        for block in self.structure:
            self._place_block(block)

    def check_stability(
        self,
        id,
        place_all_blocks=False,
        drop=False,
        position_threshold=0.01,
        rotation_threshold=0.1,
        debug=False,
    ):
        # We only want to place blocks up to the specified point
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)
        for block in self.structure:
            self._place_block(block, drop=drop)
            if (not place_all_blocks) and block.id == id:
                break

        pos1, quat1 = p.getBasePositionAndOrientation(id)

        time_step = 1.0 / 240.0  # 240 Hz simulation step
        p.setTimeStep(time_step)
        for _ in range(500):
            p.stepSimulation()
            if debug:
                time.sleep(time_step)

        pos2, quat2 = p.getBasePositionAndOrientation(id)

        pos_delta = np.array(pos2) - np.array(pos1)
        rot_delta = angular_error(quat2mat(np.array(quat2)), quat2mat(np.array(quat1)))

        pos_error = np.linalg.norm(pos_delta)
        rot_error = np.linalg.norm(rot_delta)

        stable = pos_error < position_threshold and rot_error < rotation_threshold

        return stable, pos_delta, rot_delta

    # def check_full_structure_stability(self) -> bool:
    #     """
    #     WARN: It would be more optimal to check the stability of the assembly process rather than of each block... but for our situation this is likely close enough to equivalent.
    #     stability_check in whole_structure is more appropriate
    #     """
    #     return all([
    #         self.check_stability(block.id) for block in self.structure
    #             ])


def create_block(block):
    if block.shape == "cuboid":
        return _create_cuboid(block)
    elif block.shape == "cylinder":
        return _create_cylinder(block)
    elif block.shape == "cone":
        return _create_cone(block)
    else:
        raise ValueError(f"Shape {block.shape} not supported")


def _create_cuboid(block, lateral_friction=0.5, spinning_friction=0.2):
    dimensions, position, orientation, color = (
        block.dimensions,
        block.position,
        block.orientation,
        block.color,
    )
    dimensions_m = [x / 1000 for x in dimensions]
    position_m = [x / 1000 for x in position]
    half_extents = [dim / 2 for dim in dimensions_m]

    density = 1000  # Density of the block in kg/m^3
    mass = density * dimensions_m[0] * dimensions_m[1] * dimensions_m[2]

    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    p.changeDynamics(
        id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction
    )
    return id


def _create_cylinder(block, lateral_friction=0.5, spinning_friction=0.2):
    radius, height, position, orientation, color = (
        block.dimensions[0],
        block.dimensions[1],
        block.position,
        block.orientation,
        block.color,
    )
    position_m = [x / 1000 for x in position]
    radius_m = radius / 1000
    height_m = height / 1000

    density = 1000  # Density of the block in kg/m^3
    mass = density * np.pi * radius_m**2 * height_m

    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius_m, height=height_m
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius_m, length=height_m, rgbaColor=color
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    p.changeDynamics(
        id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction
    )
    return id


import os

CONE_STL_PATH = os.path.join(os.path.dirname(__file__), "cone.stl")


def _create_cone(block, lateral_friction=0.5, spinning_friction=0.2):
    radius, height, position, orientation, color = (
        block.dimensions[0],
        block.dimensions[1],
        block.position,
        block.orientation,
        block.color,
    )
    position_m = [x / 1000 for x in position]
    radius_m = radius / 1000
    height_m = height / 1000

    density = 1000  # Density of the block in kg/m^3
    mass = density * np.pi * radius_m**2 * height_m / 3

    collision_shape = p.createCollisionShape(
        p.GEOM_MESH, fileName=CONE_STL_PATH, meshScale=[radius_m, radius_m, height_m]
    )
    visual_shape = p.createVisualShape(
        p.GEOM_MESH,
        fileName=CONE_STL_PATH,
        meshScale=[radius_m, radius_m, height_m],
        rgbaColor=color,
    )
    id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position_m,
        baseOrientation=orientation,
    )
    p.changeDynamics(
        id, -1, lateralFriction=lateral_friction, spinningFriction=spinning_friction
    )
    return id


def test_with_gui():
    if not p.isConnected():
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    structure = Structure()
    block1 = Block(
        id=1,
        block_name="block1",
        gpt_name="box",
        shape="cuboid",
        dimensions=[100, 100, 100],
        position=[0, 0, 50],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )
    block2 = Block(
        id=2,
        block_name="block2",
        gpt_name="box",
        shape="cuboid",
        dimensions=[180, 100, 180],
        position=[0, 0, 150],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )
    block3 = Block(
        id=3,
        block_name="block3",
        gpt_name="box",
        shape="cone",
        dimensions=[100, 500],
        position=[0, 0, 250],
        orientation=[0, 0, 0, 1],
        color=[1.0, 0, 0, 1.0],
    )

    structure.add_block(block1)
    structure.place_blocks()
    structure.add_block(block2)
    structure.place_blocks()
    structure.add_block(block3)
    structure.place_blocks()
    # print(structure.check_stability(2))

    # print("JSON", structure.get_json())
    # print("gpt_json", structure.get_gpt_json())
    # print("get by id", structure.get_block_by_id(1))
    # structure.place_blocks()

    # structure.get_block_by_id(1).move(
    #     [0, 20, 0], True
    # )
    # structure.place_blocks()
    # structure.get_block_by_id(1).move(
    #     [0, 50, 0], True
    # )
    # structure.place_blocks()

    # structure.delete_by_id(1)
    # structure.place_blocks()
    breakpoint()


if __name__ == "__main__":
    test_with_gui()
