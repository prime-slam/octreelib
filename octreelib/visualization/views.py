from typing import List

import vtk

from grid import GridBase
from octree import OctreeNodeBase, OctreeBase

from vtkmodules.vtkCommonColor import vtkNamedColors

__all__ = ["visualize_grid", "visualize_octree"]


def visualize_grid(grid: GridBase):
    actors = grid_actors(grid)
    run_visualization(actors)


def visualize_octree(octree: OctreeBase):
    actors = octree_actors(octree)
    run_visualization(actors)


def run_visualization(actors: List[vtk.vtkActor]):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.5, 0.6, 0.8)

    for actor in actors:
        renderer.AddActor(actor)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1600, 1200)

    # Create an interactor for user interaction
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Create an interactor style for panning and rotating the view
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Start the interactor
    interactor.Initialize()
    interactor.Start()


def octree_actors(octree: OctreeBase):
    def octree_node_actors(node: OctreeNodeBase):
        if node.has_children:
            return sum([octree_node_actors(child) for child in node.children], [])
        bounds = [
            node.bounding_box[0][0],
            node.bounding_box[1][0],
            node.bounding_box[0][1],
            node.bounding_box[1][1],
            node.bounding_box[0][2],
            node.bounding_box[1][2],
        ]
        cube_source = vtk.vtkCubeSource()
        cube_source.SetBounds(bounds)

        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube_source.GetOutputPort())

        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetRepresentationToWireframe()
        cube_actor.GetProperty().SetLineWidth(2.0)

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        for point in node.points:
            point_id = points.InsertNextPoint(point[:])
            cells.InsertNextCell(1)
            cells.InsertCellPoint(point_id)

            points.Modified()
            cells.Modified()

        input_data = vtk.vtkPolyData()
        input_data.SetPoints(points)
        input_data.SetVerts(cells)

        points_mapper = vtk.vtkPolyDataMapper()
        points_mapper.SetInputData(input_data)

        colors = vtkNamedColors()
        points_actor = vtk.vtkActor()
        points_actor.SetMapper(points_mapper)
        points_actor.GetProperty().SetColor(colors.GetColor3d("Green"))
        points_actor.GetProperty().SetPointSize(3)

        return [cube_actor, points_actor]

    return octree_node_actors(octree.root)


def grid_actors(grid: GridBase):
    return sum([octree_actors(octree) for octree in grid.octrees.values()], [])
