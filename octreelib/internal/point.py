import numpy as np

from dataclasses import dataclass

from octreelib.internal.typing import Point

__all__ = ["PointWithPose"]


# class PointWithPose(np.ndarray):
#     @classmethod
#     def from_point(cls, point: Point, pose_number: int):
#         return cls(pose_number, point.shape, point)
#
#     def __init__(self, pose_number: int, shape, buffer):
#         self.pose_number = pose_number
#         super(shape=shape, buffer=buffer)


class PointWithPose(np.ndarray):
    def __new__(cls, input_array, pose_number: int):
        obj = np.asarray(input_array).view(cls)
        obj.pose_number = pose_number
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pose_number = getattr(obj, "pose_number", None)
