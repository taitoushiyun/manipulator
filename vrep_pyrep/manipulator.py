from pyrep.robots.arms.arm import Arm
from pyrep.objects.collision import Collision
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.backend import sim
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

    def set_initial_joint_positions(self, p):
        super().set_joint_positions(p)


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

    def set_initial_joint_positions(self, p):
        super().set_joint_positions(p)


class ManipulatorCCPlane(Arm):

    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=2, collision_cnt=15):
        super().__init__(count, name, num_joints)
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]
        self.joint_index = [self._num_joints // self.num_segments * i + j
                            for i in range(self.num_segments) for j in range(2)]

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

    def set_joint_target_velocities(self, velocities: List[float]) -> None:
        for i in range(len(velocities)):
            self.joints[self.joint_index[i]].set_joint_target_velocity(velocities[i])

    def set_initial_joint_positions(self, positions: List[float], allow_force_mode=True) -> None:
        assert len(positions) == len(self.joint_index)
        if not allow_force_mode:
            for i in range(len(positions)):
                self.joints[i].set_joint_position(positions[i], allow_force_mode)
            return

        is_model = self.is_model()
        if not is_model:
            self.set_model(True)

        prior = sim.simGetModelProperty(self.get_handle())
        p = prior | sim.sim_modelproperty_not_dynamic
        # Disable the dynamics
        sim.simSetModelProperty(self._handle, p)

        for i in range(len(positions)):
            self.joints[i].set_joint_position(positions[i], allow_force_mode)
        for i in range(len(positions)):
            self.joints[i].set_joint_position(positions[i])
        sim.simExtStep(True)  # Have to step once for changes to take effect

        # Re-enable the dynamics
        sim.simSetModelProperty(self._handle, prior)
        self.set_model(is_model)


class ManipulatorCC3D(Arm):
    def __init__(self, count=0, name='manipulator', num_joints=12, num_segments=2, collision_cnt=15):
        super().__init__(count, name, num_joints)
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.agent_base = Shape('manipulator_base_visual')
        self.collisions = [Collision(cname)
                           for cname in [f'Collision{i_}' for i_ in range(collision_cnt)] + ['Collision']]
        self.joint_index = [self._num_joints // self.num_segments * i + j
                            for i in range(self.num_segments) for j in range(2)]

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

    def set_joint_target_velocities(self, velocities: List[float]) -> None:
        for i in range(len(velocities)):
            self.joints[self.joint_index[i]].set_joint_target_velocity(velocities[i])

    def set_initial_joint_positions(self, positions: List[float], allow_force_mode=True) -> None:
        assert len(positions) == len(self.joint_index)
        if not allow_force_mode:
            for i in range(len(positions)):
                self.joints[i].set_joint_position(positions[i], allow_force_mode)
            return

        is_model = self.is_model()
        if not is_model:
            self.set_model(True)

        prior = sim.simGetModelProperty(self.get_handle())
        p = prior | sim.sim_modelproperty_not_dynamic
        # Disable the dynamics
        sim.simSetModelProperty(self._handle, p)

        for i in range(len(positions)):
            self.joints[i].set_joint_position(positions[i], allow_force_mode)
        for i in range(len(positions)):
            self.joints[i].set_joint_position(positions[i])
        sim.simExtStep(True)  # Have to step once for changes to take effect

        # Re-enable the dynamics
        sim.simSetModelProperty(self._handle, prior)
        self.set_model(is_model)
