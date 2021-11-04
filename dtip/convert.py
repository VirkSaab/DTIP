import subprocess
import logging
from typing import Union
from pathlib import Path
from dtip.utils import SpinCursor


__all__ = [
    "convert_dicom_to_nifti", "fsl_to_dtitk_multi", "dtitk_to_fsl_multi"
]


def convert_raw_dicom_to_nifti(input_path: Union[str, Path],
                               output_path: Union[str, Path],
                               method: str = "dcm2nii",
                               gz: bool = True,
                               reorient: bool = True) -> int:
    """Convert raw DICOM files in `input_path` to NIfTI `.nii` or `.nii.gz` files.

    Args:
        input_path: folder path containing DICOM files of a subject.
        output_path: folder path where output files will be saved.
        method: Sometimes `dcm2niix` does not produce `.bvec` and `.bval` files
            for DTI or DWI volumes. In that case, `dcm2nii` is likely to do it.
            `auto` will use both `dcm2nii` and `dcm2niix` CLI tools to extract 
            files. It does produce a large number of files but is more robust.
            Choose one of the following conversion methods: `auto` 
            (set on auto if dcm2niix did not generate .bvecs and .bvals files)`,
            `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii). 
            [default: `dcm2nii`]
        gz: compress .nii file to .nii.gz.
        reorient: reorient the dicoms according to LAS orientation.

    Returns:
        0 on successful completion.
    """

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise NotADirectoryError("DICOM files must be in a folder.")

    if method == "auto":
        exit_code = method_dcm2nii(input_path, output_path, gz, reorient)
        x_exit_code = method_dcm2niix(input_path, output_path, gz, reorient)
        _err_msg = '[@ `convert_raw_dicom_to_nifti`] problem in auto method.'
        if exit_code + x_exit_code != 0:
            logging.error(_err_msg)
        assert exit_code + x_exit_code == 0, _err_msg
    elif method == "dcm2nii":
        exit_code = method_dcm2nii(input_path, output_path, gz, reorient)
        _err_msg = '[@ `convert_raw_dicom_to_nifti`] problem in method_dcm2nii method.'
        if exit_code != 0:
            logging.error(_err_msg)
        assert exit_code == 0, _err_msg
    elif method == "dcm2niix":
        exit_code = method_dcm2niix(input_path, output_path, gz, reorient)
        _err_msg = '[@ `convert_raw_dicom_to_nifti`] problem in method_dcm2niix method.'
        if exit_code != 0:
            logging.error(_err_msg)
        assert exit_code == 0, _err_msg
    else:
        _err_msg = f"Given {method} method not supported."
        _err_msg += "Only supports `auto`, `dcm2nii`, `dcm2niix`"
        if exit_code != 0:
            logging.error(_err_msg)
        raise NotImplementedError(_err_msg)
    return 0


def method_dcm2nii(input_path: Union[str, Path],
                   output_path: Union[str, Path],
                   gz: bool = True,
                   reorient: bool = True) -> int:
    """DICOM to NIfTI conversion using dcm2nii command.

    Args:
        input_path: folder path containing DICOM files of a subject.
        output_path: folder path where output files will be saved.
        gz: compress .nii file to .nii.gz.
        reorient: reorient the dicoms according to LAS orientation.

    Returns:
        exit_code 0 if no errors. else 1.
    """

    command = ['dcm2nii', '-4', 'Y']
    if gz:
        command += ['-g', 'Y']
    if reorient:
        command += ['-x', 'Y']
    command += ['-t', 'Y', '-d', 'N', '-o', output_path, input_path]

    with SpinCursor("dcm2nii conversion..."):
        try:
            subprocess.run(command)  # Run command
            # Get metadata in JSON files
            subprocess.run(['dcm2niix', '-b', 'o', '-o', output_path, input_path])
            return 0

        except FileNotFoundError:
            logging.error("[@ `dcm2nii`] Make sure `dcm2nii` is installed.")
            return 1


def method_dcm2niix(input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    gz: bool = True,
                    reorient: bool = True) -> int:
    """DICOM to NIfTI conversion using dcm2niix command.

    Args:
        input_path: folder path containing DICOM files of a subject.
        output_path: folder path where output files will be saved.
        gz: compress .nii file to .nii.gz.
        reorient: reorient the dicoms according to LAS orientation.

    Returns:
        exit_code 0 if no errors. else 1.
    """

    command = ["dcm2niix"]

    if gz:
        command += ['-z', 'y']
    if reorient:
        command += ['-x', 'y']
    command += ['-b', 'y', '-p', 'y', '-f', '%p_s%s', '-o', output_path, input_path]

    with SpinCursor("dcm2niix conversion..."):
        try:
            subprocess.run(command)  # Run command
            return 0

        except FileNotFoundError:
            logging.error("[@ `dcm2niix`] dcm2niix not found on system.")
            return 1
