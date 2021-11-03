import subprocess
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
        Nothing
    """

    input_path, output_path = Path(input_path), Path(output_path)

    if not input_path.is_dir():
        raise NotADirectoryError("DICOM files must be in a folder.")

    output_path.mkdir(parents=True, exist_ok=True)

    if method == "auto":
        exit_code = method_dcm2nii(input_path, output_path, gz, reorient)
        x_exit_code = method_dcm2niix(input_path, output_path, gz, reorient)
        _err_msg = '[Error @ `convert_raw_dicom_to_nifti`] problem in auto method.'
        assert exit_code + x_exit_code == 0, _err_msg
    elif method == "dcm2nii":
        exit_code = method_dcm2nii(input_path, output_path, gz, reorient)
        _err_msg = '[Error @ `convert_raw_dicom_to_nifti`] problem in method_dcm2nii method.'
        assert exit_code == 0, _err_msg
    elif method == "dcm2niix":
        exit_code = method_dcm2niix(input_path, output_path, gz, reorient)
        _err_msg = '[Error @ `convert_raw_dicom_to_nifti`] problem in method_dcm2niix method.'
        assert exit_code == 0, _err_msg
    else:
        _msg = f"Given {method} method not supported."
        _msg += "Only supports `auto`, `dcm2nii`, `dcm2niix`"
        raise NotImplementedError(_msg)
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
            return 0

        except FileNotFoundError:
            _msg = "[dcm2nii error] Make sure `dcm2nii` is installed."
            print(_msg)
            # logger.error(_msg)
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
    command += ['-p', 'y', '-f', '%p_s%s', '-o', output_path, input_path]

    with SpinCursor("dcm2niix conversion..."):
        try:
            subprocess.run(command)  # Run command
            return 0

        except FileNotFoundError:
            _msg = "[dcm2niix error] dcm2niix not found on system."
            print(_msg)
            # logger.error(_msg)
            return 1
