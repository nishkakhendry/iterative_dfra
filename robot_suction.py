import pybullet
import numpy as np
import threading

# Custom suction gripper class
class SuctionGripper:
    def __init__(self, robot, tool_link,
                 suction_range=0.05,
                 suction_force=1000):
        """
        :param robot:          bodyUniqueId of your robot base
        :param tool_link:      link index on the robot where the cup attaches
        :param suction_range:  max ray distance to detect an object (m)
        :param suction_force:  max force for holding the object (unused)
        """
        self.robot = robot
        self.tool = tool_link

        self.suction_range = suction_range
        self.suction_force = suction_force
        self.activated = False

        # # Start thread to handle additional gripper constraints.
        self.constraints_thread = threading.Thread()
        self.constraint_id = None

    # Get tool link position and orientation
    def _get_tool_pose(self):
        pos, orn = pybullet.getLinkState(self.robot, self.tool)[:2]
        return np.array(pos), orn

    # Ray-cast downward to find a nearby object under the tool.
    def _find_target(self):
        tool_pos, tool_orn = self._get_tool_pose()
        down = pybullet.rotateVector(tool_orn, [0, 0, 1])
        ray_to = tool_pos + np.array(down) * self.suction_range

        hit = pybullet.rayTest(tool_pos.tolist(), ray_to.tolist())[0]
        obj_id, link_idx, frac, hit_pos, _ = hit

        if obj_id is not None and obj_id >= 0 and obj_id != self.robot and frac < 1.0:
            return obj_id, link_idx, frac, hit_pos
        return None, None, None, None

    # Activate suction and create constraint if object is found.
    def activate(self):
        if self.activated:
            return
        obj, link, frac, hit_pos = self._find_target()
        if obj is not None:
            # compute frame offsets so the constraint attaches at the contact point
            tool_pos, tool_orn = self._get_tool_pose()
            # child_com_pos = pybullet.getLinkState(obj, link)[0]
            if link >= 0:
                # normal articulated link
                child_com_pos = pybullet.getLinkState(obj, link)[0]
            else:
                # base link hit → use base COM/origin
                child_com_pos = pybullet.getBasePositionAndOrientation(obj)[0]

            parent_frame_pos = (np.array(hit_pos) - tool_pos).tolist()
            child_frame_pos  = (np.array(hit_pos) - np.array(child_com_pos)).tolist()

            self.constraint_id = pybullet.createConstraint(
                parentBodyUniqueId=self.robot,
                parentLinkIndex=self.tool,
                childBodyUniqueId=obj,
                childLinkIndex=link,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=parent_frame_pos,
                childFramePosition=child_frame_pos,
                parentFrameOrientation=tool_orn,
                childFrameOrientation=pybullet.getBasePositionAndOrientation(obj)[1]
            )
        self.activated = True

    # Release suction by removing constraint.
    def release(self):
        if self.constraint_id is not None:
            pybullet.removeConstraint(self.constraint_id)
            self.constraint_id = None
        self.activated = False

    # Check if tool link is contacting any external object.
    def detect_contact(self):
        if not self.activated:
            return False
        pts = pybullet.getContactPoints(bodyA=self.robot, linkIndexA=self.tool)
        # filter out any contacts with the robot itself
        external = [p for p in pts if p[2] != self.robot]
        return len(external) > 0

    # Check if a given body is in contact with anything external.
    def external_contact(self, body=None):
        if body is None:
            return self.detect_contact()
        pts = pybullet.getContactPoints(bodyA=body)
        others = [p for p in pts if p[2] != self.robot]
        return len(others) > 0

    # True if constraint exists (i.e. object held).
    def check_grasp(self):
        return self.constraint_id is not None

    # Suction has no width—always returns 0.
    def grasp_width(self):
        return 0.0

    # Run ray test and return nearby object info.
    def check_proximity(self):
        obj, link, frac, _ = self._find_target()
        return obj, link, frac
