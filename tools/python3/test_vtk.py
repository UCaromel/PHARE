import sys
import vtk
import h5py
import logging
import matplotlib.pyplot as plt

from pyphare.logger import getLogger

logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__)
data_path = "phare_outputs/harris_3d/EM_B.vtkhdf"


if __name__ == "__main__":
    colors = vtk.vtkNamedColors()
    reader = vtk.vtkHDFReader(file_name=data_path)
    # reader.SetInputArrayToProcess(
    #     0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "data"
    # )

    f0 = vtk.vtkOutlineFilter()
    f0.SetInputConnection(reader.GetOutputPort())
    f0.Update()

    f1 = vtk.vtkCompositeDataGeometryFilter()
    f1.SetInputConnection(reader.GetOutputPort())
    f1.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(f1.GetOutputPort())

    actor = vtk.vtkActor(mapper=mapper)
    # actor.GetProperty().SetDiffuse(0)
    # actor.GetProperty().SetAmbient(1)
    # actor.GetProperty().SetInterpolationToFlat()
    # actor.GetProperty().SetEdgeVisibility(True)
    # actor.GetProperty().SetRepresentationToSurface()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Silver"))  # .SetBackground(0, 0, 0)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    ren_win.Render()
    iren.Start()
