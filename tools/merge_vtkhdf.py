#!/usr/bin/env python3
"""
Merge all .vtkhdf files in a directory into a single all[_suffix].vtkhdf.
Files whose name starts with 'all' are always skipped (already merged).

Two modes:
  --vds   (default)  Use HDF5 Virtual Datasets — no data copy, thin index file.
                     Requires ParaView's HDF5 to resolve relative paths correctly.
  --copy             Copy data into the merged file.  Always works in ParaView.

Both modes use the file with the most timesteps as reference for structure
(AMRBox, Steps, attrs).  Files whose timestamps do not match are skipped.
Override with --ref=<filename> to pick the reference explicitly.
Use --suffix=<suffix> to produce all_<suffix>.vtkhdf instead of all.vtkhdf.

Usage:
    python tools/merge_vtkhdf.py <output_dir> [--copy|--vds] [--ref=<file>] [--suffix=<suffix>]
    python tools/merge_vtkhdf.py phare_outputs/cfl_*/
"""

import re
import sys
import h5py
import numpy as np
from pathlib import Path


def _quantity_name(path: Path) -> str:
    """Strip model prefix: mhd_rho.vtkhdf -> rho, EM_B.vtkhdf -> B."""
    return re.sub(r"^(mhd|EM)_", "", path.stem)


def _timestamps(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["VTKHDF/Steps/Values"][:]


def _compatible_files(vtkhdf_files: list[Path], ref: Path) -> list[Path]:
    """Return files whose timestamps match ref; warn and skip others."""
    if not vtkhdf_files:
        return []
    ref_ts = _timestamps(ref)
    ok = [ref]
    for p in vtkhdf_files:
        if p == ref:
            continue
        ts = _timestamps(p)
        if np.allclose(ts, ref_ts):
            ok.append(p)
        else:
            print(
                f"  [skip] {p.name}: {len(ts)} timestamps, "
                f"expected {len(ref_ts)} matching {ref.name}"
            )
    return ok


def _write_structure(out: h5py.File, ref_path: Path, levels: list[str]) -> None:
    """Write VTKHDF structural data (attrs, AMRBox, Steps) from ref_path into out."""
    with h5py.File(ref_path, "r") as ref:
        ref_vtk = ref["VTKHDF"]
        vtk = out["VTKHDF"]

        for k, v in ref_vtk.attrs.items():
            vtk.attrs[k] = v

        for lvl in levels:
            ref_lvl = ref_vtk[lvl]
            g = vtk.create_group(lvl)
            for k, v in ref_lvl.attrs.items():
                g.attrs[k] = v
            g.create_dataset("AMRBox", data=ref_lvl["AMRBox"][:])
            g.create_group("CellData")
            g.create_group("FieldData")
            g.create_group("PointData")

        ref_steps = ref_vtk["Steps"]
        steps = vtk.create_group("Steps")
        for k, v in ref_steps.attrs.items():
            steps.attrs[k] = v
        steps.create_dataset("Values", data=ref_steps["Values"][:])

        for lvl in levels:
            ref_ls = ref_steps[lvl]
            ls = steps.create_group(lvl)
            ls.create_dataset("AMRBoxOffset", data=ref_ls["AMRBoxOffset"][:])
            ls.create_dataset("NumberOfAMRBox", data=ref_ls["NumberOfAMRBox"][:])
            ls.create_group("CellDataOffset")
            ls.create_group("FieldDataOffset")
            ls.create_group("PointDataOffset")


def _merge_copy(output_dir: Path, files: list[Path], levels: list[str], ref_path: Path, out_name: str) -> None:
    """Merge by copying data into out_name."""
    out_path = output_dir / out_name
    with h5py.File(out_path, "w") as out:
        out.create_group("VTKHDF")
        _write_structure(out, ref_path, levels)

        for vtkf in files:
            name = _quantity_name(vtkf)
            print(f"  copying {vtkf.name} -> {name} ...", end=" ", flush=True)
            with h5py.File(vtkf, "r") as src:
                for lvl in levels:
                    src_pd = src[f"VTKHDF/{lvl}/PointData"]
                    if "data" not in src_pd:
                        continue
                    out[f"VTKHDF/{lvl}/PointData"].create_dataset(
                        name,
                        data=src_pd["data"][:],
                        compression="gzip",
                        compression_opts=1,
                    )
                    offset_path = f"VTKHDF/Steps/{lvl}/PointDataOffset"
                    src_off = src[f"VTKHDF/Steps/{lvl}/PointDataOffset/data"]
                    if name not in out[offset_path]:
                        out[offset_path].create_dataset(name, data=src_off[:])
            print("done")

    print(f"[ok] {out_path}")


def _merge_vds(output_dir: Path, files: list[Path], levels: list[str], ref_path: Path, out_name: str) -> None:
    """Merge using HDF5 Virtual Datasets — no data copy."""
    out_path = output_dir / out_name
    with h5py.File(out_path, "w") as out:
        out.create_group("VTKHDF")
        _write_structure(out, ref_path, levels)

    with h5py.File(out_path, "a") as out:
        for vtkf in files:
            name = _quantity_name(vtkf)
            rel = vtkf.name  # resolved relative to out_path's directory by HDF5
            print(f"  linking {vtkf.name} -> {name} ...", end=" ", flush=True)
            with h5py.File(vtkf, "r") as src:
                for lvl in levels:
                    src_pd = src[f"VTKHDF/{lvl}/PointData"]
                    if "data" not in src_pd:
                        continue
                    src_ds = src_pd["data"]
                    layout = h5py.VirtualLayout(shape=src_ds.shape, dtype=src_ds.dtype)
                    layout[...] = h5py.VirtualSource(
                        rel, f"VTKHDF/{lvl}/PointData/data", shape=src_ds.shape
                    )
                    out[f"VTKHDF/{lvl}/PointData"].create_virtual_dataset(name, layout)

                    offset_path = f"VTKHDF/Steps/{lvl}/PointDataOffset"
                    src_off = src[f"VTKHDF/Steps/{lvl}/PointDataOffset/data"]
                    if name not in out[offset_path]:
                        out[offset_path].create_dataset(name, data=src_off[:])
            print("done")

    print(f"[ok] {out_path}")


def merge(output_dir: Path, use_vds: bool = True, ref_name: str | None = None, suffix: str | None = None) -> None:
    out_name = f"all_{suffix}.vtkhdf" if suffix else "all.vtkhdf"
    all_files = sorted(p for p in output_dir.glob("*.vtkhdf") if not p.stem.startswith("all"))
    if not all_files:
        print(f"[skip] no .vtkhdf files in {output_dir}")
        return

    if ref_name is not None:
        ref_path = output_dir / ref_name
        if ref_path not in all_files:
            print(f"[error] --ref={ref_name} not found in {output_dir}")
            return
    else:
        ref_path = max(all_files, key=lambda p: len(_timestamps(p)))

    files = _compatible_files(all_files, ref_path)
    if not files:
        print(f"[skip] no compatible files in {output_dir}")
        return

    with h5py.File(ref_path, "r") as ref:
        levels = sorted(k for k in ref["VTKHDF"].keys() if k.startswith("Level"))

    mode = "vds" if use_vds else "copy"
    print(f"{output_dir}  [{mode}, {len(files)} files, {levels}]")

    if use_vds:
        _merge_vds(output_dir, files, levels, ref_path, out_name)
    else:
        _merge_copy(output_dir, files, levels, ref_path, out_name)


def main() -> None:
    use_vds = "--copy" not in sys.argv
    ref_name = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--ref=")), None)
    suffix = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--suffix=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    dirs = [Path(a) for a in args] if args else [Path(".")]
    for d in dirs:
        if not d.is_dir():
            print(f"[skip] {d} is not a directory")
            continue
        merge(d, use_vds=use_vds, ref_name=ref_name, suffix=suffix)


if __name__ == "__main__":
    main()
