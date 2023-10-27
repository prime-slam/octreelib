from typing import Annotated, Literal, Iterable

import numpy as np
import numpy.typing as npt


__all__ = [
    "Point",
    "PointCloud",
    "PosePoint",
    "PosePointCloud",
    "CPoint",
    "CPointCloud",
    "CPosePoint",
    "CPosePointCloud",
]


Point = Annotated[npt.NDArray[np.float_], Literal[3]]
PointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]
PosePoint = Annotated[npt.NDArray[np.float_], Literal[4]]
PosePointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 4]]


"""
These classes are subclasses of np.ndarray
https://numpy.org/doc/stable/user/basics.subclassing.html
"""


class CPoint(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_pose(self, pose_number: int):
        new_array = np.hstack([self, np.array([pose_number])])
        return CPosePoint(new_array)

    def to_hashable(self):
        return self.tolist()


class CPointCloud(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_poses(self, pose_numbers: Iterable[int]):
        new_array = np.hstack([self, np.array(pose_numbers).reshape((len(self), 1))])
        return CPosePointCloud(new_array)

    def with_pose(self, pose_numbers: int):
        return self.with_poses([pose_numbers] * len(self))

    def __iter__(self) -> CPoint:
        for value in super().__iter__():
            yield CPoint(value)

    def __getitem__(self, item) -> CPoint:
        return CPoint(super().__getitem__(item))

    def extend(self, other):
        return CPointCloud(np.vstack((self, other)))


class CPosePoint(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_pose(self):
        return CPoint(self[:3])

    def pose(self):
        return self[3]


class CPosePointCloud(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_poses(self):
        return CPointCloud(self[:, :3])

    def poses(self):
        return self[:, 3]

    def __iter__(self):
        for value in super().__iter__():
            yield CPosePoint(value)

    def __getitem__(self, item):
        return CPosePoint(super().__getitem__(item))

    def copy(self):
        return CPosePointCloud(super().copy())

    def extend(self, other):
        return CPosePointCloud(np.vstack((self, other)))
