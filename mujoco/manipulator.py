

class ManipulatorPlane:
    def __init__(self, sim=None, num_joints=12, num_segments=2, collision_cnt=15):
        self.sim = sim
        self.joint_index = range(1, num_joints, 2)

    def get_collision_result(self):
        pass

    def get_joint_positions(self):
        return self.sim.data.qpos[self.joint_index]

    def get_joint_velocities(self):
        return self.sim.data.qvel[self.joint_index]

    def set_joint_target_velocities(self, velocities):
        for i in range(len(velocities)):
            self.sim.data.ctrl[i] = velocities[i]

    def set_initial_joint_positions(self, positions, allow_force_mode):
        pass


class Manipulator3D:
    def __init__(self, sim=None, num_joints=12, num_segments=6, collision_cnt=15):
        self.sim = sim
        self.joint_index = list(range(num_joints))
        assert len(self.sim.data.ctrl) == num_joints

    def get_collision_result(self):
        pass

    def get_joint_positions(self):
        return self.sim.data.qpos

    def get_joint_velocities(self):
        return self.sim.data.qvel

    def set_joint_target_velocities(self, velocities):
        for i in range(len(velocities)):
            self.sim.data.ctrl[i] = velocities[i]

    def set_initial_joint_positions(self, positions, allow_force_mode):
        pass


class ManipulatorCCPlane:

    def __init__(self, sim, num_joints=12, num_segments=2, collision_cnt=15):
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.joint_index = [self._num_joints // self.num_segments * i + j
                            for i in range(self.num_segments) for j in range(2)]

    def get_collision_result(self):
        pass

    def get_base(self):
        pass

    def get_joint_initial_positions(self):
        pass

    def get_joint_positions(self):
        pass

    def get_joint_velocities(self):
        pass

    def set_joint_target_velocities(self, velocities):
        pass

    def set_initial_joint_positions(self, positions, allow_force_mode):
        pass


class ManipulatorCC3D:
    def __init__(self, sim, num_joints=12, num_segments=2, collision_cnt=15):
        assert num_joints % (2 * num_segments) == 0, 'num_joints and num_segments not matching'
        self.num_segments = num_segments
        self.joint_index = [self._num_joints // self.num_segments * i + j
                            for i in range(self.num_segments) for j in range(2)]

    def get_collision_result(self):
        pass

    def get_base(self):
        pass

    def get_joint_initial_positions(self):
        pass

    def get_joint_positions(self):
        pass

    def get_joint_velocities(self):
        pass

    def set_joint_target_velocities(self, velocities):
        pass

    def set_initial_joint_positions(self, positions, allow_force_mode):
        pass