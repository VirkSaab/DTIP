import logging
import shutil
import nibabel as nib
from typing import Union
from pathlib import Path
from dtip.utils import show_exec_time

__all__ = ["locate_dti_files"]

@show_exec_time
def locate_dti_files(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    protocol_names: list,
    n_gradients: int,
    ret_paths: bool = True,
) -> Union[dict, int]:
    """Locate DTI-related data and metadata file

    Args:
        input_path: folder containing DTI files.
        output_path: copy located files (if any) to `output_path` 
            and rename as `dtidata.<extension>.`
        ret_paths: if True, return dict with selected files paths where
            key is file extension and value is the file path.

    Returns:
        if ret_paths is True, return dict with selected files paths where
        key is file extension and value is the file path. 
        Otherwise, return 0 on successful execution.
    """

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # First, look for .bvec and .bval files and get the corresponding DTI file
    bval_paths = sorted([p for p in input_path.glob("*") if p.suffix == ".bval"])
    bvec_paths = sorted([p for p in input_path.glob("*") if p.suffix == ".bvec"])

    _err_msg = f"found {len(bval_paths)} .bval files and {len(bvec_paths)} .bval files"
    assert len(bval_paths) == len(bvec_paths), _err_msg

    if len(bvec_paths) == 0:
        _errmsg = "No .bvec file found. Cannot proceed without .bvec file."
        logging.error(_errmsg)
        raise FileNotFoundError(_errmsg)

    # Make protocol and series based pairs
    dti_series_dict = {}
    for protocol_name in protocol_names:
        for bval_filename, bvec_filename in zip(bval_paths, bvec_paths):
            _errmsg = ".bval and .bvec have different filenames."
            assert bvec_filename.stem == bval_filename.stem, _errmsg

            if protocol_name in bvec_filename.stem:
                dti_series_dict[bvec_filename.stem] = {
                    "bvec": bvec_filename,
                    "bval": bval_filename,
                }

    # Add DTI nifti files to series
    for name, paths in dti_series_dict.items():
        for p in input_path.glob("*"):
            if (str(p).endswith(".nii.gz")) and (p.stem == f"{name}.nii"):
                # Check if the DTI volume has required gradients directions
                if nib.load(p).shape[3] == n_gradients:
                    paths.update({"nifti": p})

    # Keep series with all the required files
    dti_series_dict = {  # series at least have ['bvec', 'bval', 'nifti']
        k: v for k, v in dti_series_dict.items() if len(v) >= 3
    }

    if len(dti_series_dict) == 0:
        _errmsg = "No matching DTI series found."
        logging.error(_errmsg)
        raise FileNotFoundError(_errmsg)

    if ret_paths:
        return dti_series_dict

    # Rest of the series are same. Use anyone of them
    if len(dti_series_dict) > 1:
        _msg = f"Located multiple matching series. \n{dti_series_dict.keys()}"
        logging.info(_msg)
    else:
        logging.info(f"Matched series = {dti_series_dict.keys()}")

    for series_name, selected_paths in dti_series_dict.items():
        if protocol_name[0] in series_name:
            break

    # Get related metadata JSON file
    if "DTImediumiso" in series_name:
        series, acq = series_name.replace("xDTImediumiso", "")[1:].split("a")
        selected_json = [
            p
            for p in input_path.glob("*")
            if (("DTI_medium_iso" in p.stem) and (p.suffix == ".json"))
        ]
        if len(selected_json) > 0:
            selected_json = [
                p for p in selected_json if p.stem.split("_")[-1] == series
            ]
    else:
        _series = series_name.split("_")[-1]
        selected_json = [
            p
            for p in input_path.glob("*")
            if (("DTI_medium_iso" in p.stem) and (p.suffix == ".json"))
        ]
        if len(selected_json) > 0:
            selected_json = [
                p for p in selected_json if p.stem.split("_")[-1] == _series
            ]
    if len(selected_json) > 0:  # Add the selected JSON to selected_paths
        selected_paths["json"] = selected_json[0]

    # Copy selected files to the output folder
    for name, src in selected_paths.items():
        dst = output_path / src.name
        shutil.copy(src, dst)
        logging.info(f"Copied {name} @ {dst}")
        selected_paths[name] = dst

    return 0, selected_paths
