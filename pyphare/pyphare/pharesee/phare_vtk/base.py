#
#
#

import vtk


class VtkFile:
    def __init__(self, filename, array_name="data"):
        self.array_name = array_name
        self.reader = vtk.vtkHDFReader(file_name=filename)

        self.surface = vtk.vtkDataSetSurfaceFilter()
        self.surface.SetInputConnection(self.reader.GetOutputPort())
        self.surface.Update()

        self.geom = vtk.vtkCompositeDataGeometryFilter()
        self.geom.SetInputConnection(self.surface.GetOutputPort())
        self.geom.Update()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.geom.GetOutputPort())
        self.mapper.SelectColorArray(array_name)
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.SetScalarRange(
            self.geom.GetOutput().GetPointData().GetArray(array_name).GetRange(0)
        )


class VtkFieldFile(VtkFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VtkTensorFieldFile(VtkFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
