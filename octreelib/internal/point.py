import numpy as np

from dataclasses import dataclass

from octreelib.internal.typing import Point

__all__ = ["PointWithPose"]


class PointWithPose(np.ndarray):
    """
    This class is a subclass of np.ndarray which has one additional field: pose_number: int
    https://numpy.org/doc/stable/user/basics.subclassing.html
    """

    def __new__(cls, input_array, pose_number: int):
        obj = np.asarray(input_array).view(cls)
        obj.pose_number = pose_number
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pose_number = getattr(obj, "pose_number", None)
