import numpy as np
import math


def dh_matrix(d, a, theta, alpha):

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    return np.array([[cos_theta, -cos_alpha * sin_theta, sin_alpha * sin_theta, a * cos_theta],
                    [sin_theta, cos_alpha * cos_theta, -sin_alpha * cos_theta, a * sin_theta],
                    [0, sin_alpha, cos_alpha, d], [0, 0, 0, 1]])


def dh_table(theta, l):
    ang = math.radians(90)
    return [[0, 0, theta, -ang],
            [0, l, theta, ang]]


class DHModel(object):
    def __init__(self, model_index, num_joints):
        self.model_index = model_index
        self.num_joints = num_joints

    def forward_kinematics(self, theta):
        assert len(theta) % 2 == 0 and len(theta) == self.num_joints
        L = 0.1
        temp = np.array([[1, 0, 0, 0.2],
                        [0, 1, 0, -self.model_index],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]])
        for i in range(len(theta)):
            temp = temp.dot(dh_matrix(*dh_table(theta[i], L)[i % 2]))
        ee_point = temp[0:3, 3].flatten()
        return ee_point

