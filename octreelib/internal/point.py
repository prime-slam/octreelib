from typing import Annotated, Literal

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


# def ensure_no_pose(point: Union[Point, PosePoint]) -> Point:
#     return point if len(point) == 3 else remove_pose_from_point(point)


# class PointWithPose(np.ndarray):
#     """
#     This class is a subclass of np.ndarray which has one additional field: pose_number: int
#     https://numpy.org/doc/stable/user/basics.subclassing.html
#     """
#
#     def __new__(cls, input_array, pose_number: int):
#         obj = np.asarray(input_array).view(cls)
#         obj.pose_number = pose_number
#         return obj
#
#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.pose_number = getattr(obj, "pose_number", None)
#
#
# def add_clouds(point_cloud_a: PointWithPose, point_cloud_b: PointWithPose):
#     if point_cloud_a.pose_number != point_cloud_b.pose_number:
#         raise ValueError("pose numbers of the point clouds must be the same")
#     return np.vstack([point_cloud_a, point_cloud_b])
