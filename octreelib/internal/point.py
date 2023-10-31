from typing import Annotated, Literal, Iterable

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

RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


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


class Point(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_pose(self, pose_number: int):
        """
        :param pose_number: New pose number.
        :return: The same point but with information about pose number.
        """
        new_array = np.hstack([self, np.array([pose_number])])
        return PosePoint(new_array)


class PointCloud(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def with_poses(self, pose_numbers: Iterable[int]):
        """
        :param pose_numbers: New pose numbers.
        :return: The same point cloud but with information about pose numbers.
        """
        new_array = np.hstack([self, np.array(pose_numbers).reshape((len(self), 1))])
        return PosePointCloud(new_array)

    def with_pose(self, pose_numbers: int):
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

    def copy(self):
        """
        :return: A copy of this object.
        """
        return PointCloud(super().copy())

    def extend(self, other):
        """
        Add two point clouds
        :param other: Other PosePointCloud
        :return: Concatenation of self and other.
        """
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PointCloud(np.vstack((self, other)))


class PosePoint(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_pose(self):
        """
        :return: The same point but without information about pose numbers.
        """
        return Point(self[:3])

    def pose(self):
        """
        :return: Pose number.
        """
        return self[3]


class PosePointCloud(np.ndarray):
    def __new__(cls, input_array):
        # Construct from existing np.ndarray.
        obj = np.asarray(input_array).view(cls)
        return obj

    def without_poses(self):
        """
        :return: The same point cloud but without information about pose numbers.
        """
        return PointCloud(self[:, :3])

    def poses(self):
        """
        :return: Pose numbers.
        """
        return self[:, 3]

    def __iter__(self):
        # When iterating, return a Point class instead of a np.ndarray.
        for value in super().__iter__():
            yield PosePoint(value)

    def __getitem__(self, index):
        # When getting item by index, return a Point class instead of a np.ndarray.
        return PosePoint(super().__getitem__(index))

    def copy(self):
        """
        :return: A copy of this object.
        """
        return PosePointCloud(super().copy())

    def extend(self, other):
        """
        Add two point clouds
        :param other: Other PosePointCloud
        :return: Concatenation of self and other.
        """
        # Cannot override __add__ because it is used for different
        # purposes in parent np.ndarray.
        return PosePointCloud(np.vstack((self, other)))
