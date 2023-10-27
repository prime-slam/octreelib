from typing import Annotated, Literal, Iterable

import numpy as np
import numpy.typing as npt


__all__ = [
    "Point",
    "PointCloud",
    "PosePoint",
    "PosePointCloud",
    "add_pose_to_point",
    "add_pose_to_point_cloud",
    "remove_pose_from_point",
    "remove_pose_from_point_cloud",
    # "PointWithPose",
    "CPoint",
    "CPointCloud",
    "CPosePoint",
    "CPosePointCloud",
]


Point = Annotated[npt.NDArray[np.float_], Literal[3]]
PointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]
PosePoint = Annotated[npt.NDArray[np.float_], Literal[4]]
PosePointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 4]]


def add_pose_to_point_cloud(
    point_cloud: PointCloud, pose_number: int
) -> PosePointCloud:
    if len(point_cloud.shape) < 2:
        raise ValueError("use this method with point clouds (np.ndarray[Nx3])")
    return np.hstack([point_cloud, np.array([[pose_number]] * point_cloud.shape[0])])


def add_pose_to_point(point: Point, pose_number: int) -> PosePoint:
    if len(point.shape) != 1:
        raise ValueError("use this method with points (np.ndarray[3])")
    return np.hstack([point, np.array([pose_number])])


def remove_pose_from_point_cloud(point_cloud: PosePointCloud) -> PointCloud:
    if len(point_cloud.shape) < 2:
        raise ValueError("use this method with point clouds (np.ndarray[Nx3])")
    return point_cloud[:, :3]


def remove_pose_from_point(point: PosePoint) -> Point:
    if len(point.shape) != 1:
        raise ValueError("use this method with points (np.ndarray[3])")
    return point[:3]


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

    def with_pose(self, pose_numbers: int):
        new_array = np.hstack([self, np.array([[pose_numbers]] * len(self))])
        return CPosePointCloud(new_array)

    def with_poses(self, pose_numbers: Iterable[int]):
        new_array = np.hstack([self, np.array(np.array(pose_numbers))])
        return CPosePointCloud(new_array)

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
