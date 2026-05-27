from paraview.simple import *

from importlib import import_module

_w = None
for _qt in ("PyQt5", "PySide2", "PySide6"):
    try:
        _w = import_module(f"{_qt}.QtWidgets")
        break
    except ImportError:
        continue
if _w is None:
    raise ImportError("No Qt bindings found (tried PyQt5, PySide2, PySide6)")

# pip Qt binding links its own Qt lib, separate from ParaView's internal Qt,
# so QApplication.instance() is None here -> must create one before any QWidget.
_app = _w.QApplication.instance() or _w.QApplication([])
QInputDialog = _w.QInputDialog


def _ask_float(label, default):
    val, ok = QInputDialog.getDouble(None, "Disk parameters", label, default, -1e9, 1e9, 4)
    if not ok:
        raise RuntimeError("cancelled")
    return val


# prompts
radius = _ask_float("Outer radius:", 1.0)
tx = _ask_float("Translation X:", 0.0)
ty = _ask_float("Translation Y:", 0.0)
tz = _ask_float("Translation Z:", 0.0)

# disk source
disk = Disk()
disk.CircumferentialResolution = 50
disk.InnerRadius = 0.0
disk.OuterRadius = radius
# RadialResolution and others stay default

# translation via representation "Transforming > Translation"
rep = Show(disk)
rep.Translation = [tx, ty, tz]
Render()
