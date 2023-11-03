from __future__ import annotations

from typing import Annotated, Literal, Iterable, Tuple

import numpy as np
import numpy.typing as npt


__all__ = [
    "RawPoint",
    "RawPointCloud",
    "Point",
    "PointCloud",
    "PosePoint",
    "PosePointCloud",
]

"""
RawPoint and RawPointCloud are intended to be used in the methods
which interact with the user or the methods which facilitate those.
These are meant to be the main types for Points and Point Clouds to
be used by user. 

The classes Point, PosePoint, PointCloud, PosePointCloud are intended
for internal use. They introduce dichotomy for having and not having
information about pose number. These classes are subclasses of numpy
and are implemented according to
https://numpy.org/doc/stable/user/basics.subclassing.html
"""

RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


class Point(np.ndarray):
    """
    Represents a point. Designed for internal use.
    """

    def __new__(cls, input_array):
        """
        Construct from existing array.
        :param input_array: existing array.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_pose(self, pose_number: int):
        """
        :param pose_number: New pose number.
        :return: The same point but with information about pose number.
        """
        new_array = np.hstack([self, np.array([pose_number])])
        return PosePoint(new_array)

    def __hash__(self):
        return hash((float(self[0]), float(self[1]), float(self[2])))


class PointCloud(np.ndarray):
    """
    Represents a point cloud. Designed for internal use.
    """

    def __new__(cls, input_array):
        """
        Construct from existing array.
        :param input_array: existing array.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 3), dtype=float))

    def with_poses(self, pose_numbers: Iterable[int]) -> PosePointCloud:
        """
        :param pose_numbers: New pose numbers.
        :return: The same point cloud but with information about pose numbers.
        """
        new_array = np.hstack([self, np.array(pose_numbers).reshape((len(self), 1))])
        return PosePointCloud(new_array)

    def with_pose(self, pose_numbers: int) -> PosePointCloud:
        """
        :param pose_numbers: New pose number.
        :return: The same point cloud but with information about pose numbers.
        """
        return self.with_poses([pose_numbers] * len(self))

    def __iter__(self) -> Point:
        # when iterating, return a Point class instead of a np.ndarray.
        for value in super().__iter__():
            yield Point(value)

    def __getitem__(self, index) -> Point:
        # getting item by index, return a Point class instead of a np.ndarray.
        return Point(super().__getitem__(index))

    def copy(self) -> PointCloud:
        """
        :return: A copy of this object.
        """
        return PointCloud(super().copy())

    def extend(self, other: PointCloud) -> PointCloud:
        """
        Add two point clouds
        :param other: Other PosePointCloud
        :return: Concatenation of self and other.
        """
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PointCloud(np.vstack((self, other)))

    def __hash__(self):
        return hash(self.tobytes())


class PosePoint(np.ndarray):
    """
    Represents a point with pose. Designed for internal use.
    """

    def __new__(cls, input_array):
        """
        Construct from existing array.
        :param input_array: existing array.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_pose(self) -> Point:
        """
        :return: The same point but without information about pose numbers.
        """
        return Point(self[:3])

    def __hash__(self):
        return hash((float(self[0]), float(self[1]), float(self[2])))

    def pose(self):
        """
        :return: Pose number.
        """
        return self[3]


class PosePointCloud(np.ndarray):
    """
    Represents a point cloud with poses. Designed for internal use.
    """

    def __new__(cls, input_array):
        """
        Construct from existing array.
        :param input_array: existing array.
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 4), dtype=float))

    def without_poses(self) -> PointCloud:
        """
        :return: The same point cloud but without information about pose numbers.
        """
        return PointCloud(self[:, :3])

    def poses(self):
        """
        :return: Pose numbers.
        """
        return self[:, 3]

    def __iter__(self) -> PosePoint:
        # When iterating, return a Point class instead of a np.ndarray.
        for value in super().__iter__():
            yield PosePoint(value)

    def __getitem__(self, index) -> PosePoint:
        # When getting item by index, return a Point class instead of a np.ndarray.
        return PosePoint(super().__getitem__(index))

    def copy(self) -> PosePointCloud:
        """
        :return: A copy of this object.
        """
        return PosePointCloud(super().copy())

    def extend(self, other: PosePointCloud) -> PosePointCloud:
        """
        Add two point clouds
        :param other: Other PosePointCloud
        :return: Concatenation of self and other.
        """
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PosePointCloud(np.vstack((self, other)))

    def filtered_by_pose(self, pose_number: int) -> PosePointCloud:
        """
        :param pose_number: Pose number.
        :return: A point cloud, where each point is related to the desired pose number.
        """
        return PosePointCloud(self[self[:, 3] == pose_number])

    def __hash__(self):
        return hash(self.tobytes())
