import sys
import vtk
import h5py
import logging
import matplotlib.pyplot as plt

from pyphare.logger import getLogger

logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__)
data_path = "/home/p/git/phare/master/phare_outputs/harris_3d/EM_B.vtkhdf"
# data_path = "phare_outputs/vtk_diagnostic_test/test_vtk/test_dump_diags_1/1/2/1/ions_pop_alpha_flux.vtkhdf"
array_name = "data"

if __name__ == "__main__":
    from pyphare.pharesee.phare_vtk import show as vtk_show
    from pyphare.pharesee.phare_vtk import plot as vtk_plot

    # vtk_plot(data_path)
    vtk_show(data_path)

else:
    colors = vtk.vtkNamedColors()
    reader = vtk.vtkHDFReader(file_name=data_path)
    # reader.SetInputArrayToProcess(
    #     0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "data"
    # )

    # f0 = vtk.vtkOutlineFilter()
    # f0.SetInputConnection(reader.GetOutputPort())
    # f0.Update()

    # plane = vtk.vtkPlane()
    # plane.SetNormal(0, 0, 1)
    # plane.SetOrigin(0, 0, 0)

    # cutter = vtk.vtkCutter()
    # cutter.SetCutFunction(plane)
    # cutter.SetInputConnection(reader.GetOutputPort())

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    surface.Update()

    print("surface", surface)
    print("surface.GetOutput", surface.GetOutput())
    # poly = surface.GetOutput()

    # cell_data = poly.GetCellData()
    # array_name = cell_data.GetArrayName(0)  # or explicit name
    # cell_data.SetActiveScalars(array_name)

    # calc = vtk.vtkArrayCalculator()
    # calc.SetInputConnection(surface.GetOutputPort())
    # calc.AddVectorArrayName(array_name)
    # calc.SetFunction(f"mag({array_name})")
    # calc.SetResultArrayName("magnitude")
    # calc.Update()

    geom = vtk.vtkCompositeDataGeometryFilter()
    geom.SetInputConnection(surface.GetOutputPort())
    geom.Update()

    poly = geom.GetOutput()
    print(poly.GetNumberOfCells(), poly.GetNumberOfPoints())

    pd = poly.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        print(pd.GetArrayName(i))

    print(poly.GetPointData().GetArray(array_name).GetRange())

    pd.SetActiveScalars("data")

    vec_array = pd.GetArray(array_name)
    print("Number of components:", vec_array.GetNumberOfComponents())

    # calc = vtk.vtkArrayCalculator()
    # calc.SetInputConnection(geom.GetOutputPort())
    # calc.AddVectorArrayName(array_name)
    # calc.SetFunction(f"mag({array_name})")
    # calc.SetResultArrayName("mag")
    # calc.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(geom.GetOutputPort())
    mapper.SelectColorArray("data")
    mapper.SetScalarModeToUsePointData()
    mapper.SetScalarRange(geom.GetOutput().GetPointData().GetArray("data").GetRange(0))
    # mapper.SetScalarRange(calc.GetOutput().GetPointData().GetArray("mag").GetRange())

    actor = vtk.vtkActor(mapper=mapper)
    # actor.GetProperty().SetDiffuse(0)
    # actor.GetProperty().SetAmbient(1)
    # actor.GetProperty().SetInterpolationToFlat()
    # actor.GetProperty().SetEdgeVisibility(True)
    actor.GetProperty().SetRepresentationToSurface()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Silver"))  # .SetBackground(0, 0, 0)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(800, 600)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    ren_win.Render()
    # iren.Start()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("phare_surface.png")
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
