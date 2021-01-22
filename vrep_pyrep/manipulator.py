from pyrep.robots.arms.arm import Arm
from pyrep.objects.collision import Collision
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from typing import List, Tuple


class ManipulatorPlane(Arm):
    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=6, collision_cnt=15):
        super().__init__(count, name, num_joints)
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]

    def get_collision_result(self) -> List:
        return [c.read_collision() for c in self.collisions]

    def get_base(self) -> Shape:
        return self.agent_base

    def get_joint_initial_positions(self) -> List[float]:
        return super().get_joint_positions()

    def get_joint_positions(self) -> List[float]:
        return [self.joints[i].get_joint_position() for i in range(len(self.joints)) if i % 2 != 0]

    def get_joint_velocities(self) -> List[float]:
        return [self.joints[i].get_joint_velocity() for i in range(len(self.joints)) if i % 2 != 0]


class Manipulator3D(Arm):
    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=6, collision_cnt=15):
        super().__init__(count, name, num_joints)
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]

    def get_collision_result(self) -> List:
        return [c.read_collision() for c in self.collisions]

    def get_base(self) -> Shape:
        return self.agent_base

    def get_joint_initial_positions(self) -> List[float]:
        return super().get_joint_positions()


class ManipulatorCCPlane(Arm):

    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=2, collision_cnt=15):
        super().__init__(count, name, num_joints)
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]

    def get_collision_result(self) -> List:
        return [c.read_collision() for c in self.collisions]

    def get_base(self) -> Shape:
        return self.agent_base

    def get_joint_initial_positions(self) -> List[float]:
        return super().get_joint_positions()

    def get_joint_positions(self) -> List[float]:
        return [self.joints[self._num_joints // self.num_segments * i].get_joint_position()
                for i in range(self.num_segments)]

    def get_joint_velocities(self) -> List[float]:
        return [self.joints[self._num_joints // self.num_segments * i].get_joint_velocity()
                for i in range(self.num_segments)]

    def set_joint_target_velocities(self):
        pass


class ManipulatorCC3D(Arm):
    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=2, collision_cnt=15):
        super().__init__(count, name, num_joints)
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]

    def get_collision_result(self) -> List:
        return [c.read_collision() for c in self.collisions]

    def get_base(self) -> Shape:
        return self.agent_base

    def get_joint_initial_positions(self) -> List[float]:
        return super().get_joint_positions()

    def get_joint_positions(self) -> List[float]:
        return [self.joints[self._num_joints // self.num_segments * i + j].get_joint_position()
                for i in range(self.num_segments) for j in range(2)]

    def get_joint_velocities(self) -> List[float]:
        return [self.joints[self._num_joints // self.num_segments * i + j].get_joint_velocity()
                for i in range(self.num_segments) for j in range(2)]

    def set_joint_target_velocities(self):
        pass

