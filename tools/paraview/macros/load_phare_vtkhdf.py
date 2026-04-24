"""
ParaView macro — select multiple PHARE .vtkhdf files, verify they share the
same timestamps, and merge them into a single source with named arrays.

Save as macro: Tools -> Macros -> Add Macro -> select this file.
"""

import os
import re

from paraview.simple import (
    AppendAttributes,
    GetAnimationScene,
    OpenDataFile,
    RenameArrays,
    Render,
    Show,
)

from importlib import import_module
for _qt in ("PyQt5", "PySide2", "PySide6"):
    try:
        _m = import_module(f"{_qt}.QtWidgets")
        QApplication, QFileDialog = _m.QApplication, _m.QFileDialog
        break
    except ImportError:
        continue
else:
    raise ImportError("No Qt bindings found (tried PyQt5, PySide2, PySide6)")

import h5py
import numpy as np


def _quantity_name(fpath):
    fname = os.path.basename(fpath)
    return re.sub(r"^(mhd|EM)_", "", os.path.splitext(fname)[0])


def _timestamps(fpath):
    with h5py.File(fpath, "r") as f:
        return f["VTKHDF/Steps/Values"][:]


def _run():
    app = QApplication.instance() or QApplication([])
    paths, _ = QFileDialog.getOpenFileNames(
        None, "Select PHARE .vtkhdf files", "", "VTKHDF files (*.vtkhdf)"
    )
    if not paths:
        print("[cancelled] no files selected.")
        return

    ref_path = max(paths, key=lambda p: len(_timestamps(p)))
    ref_ts = _timestamps(ref_path)

    for fpath in paths:
        ts = _timestamps(fpath)
        if not np.allclose(ts, ref_ts):
            raise RuntimeError(
                f"{os.path.basename(fpath)} has {len(ts)} timestamps, "
                f"expected {len(ref_ts)} matching {os.path.basename(ref_path)}"
            )

    sources = []
    for fpath in paths:
        name = _quantity_name(fpath)
        reader = OpenDataFile(fpath)
        rename = RenameArrays(Input=reader)
        rename.PointArrays = ["data", name]
        sources.append(rename)
        print(f"[loaded] {os.path.basename(fpath)} -> {name}")

    merged = AppendAttributes(Input=sources)
    Show(merged)
    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
    Render()
    print(f"[ok] merged {len(sources)} quantities")


_run()
