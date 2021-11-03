"""This module implements the DTI preprocessing pipeline for one and multi
    subject DTI data using FSL in majority.

    Source: https://open.win.ox.ac.uk/pages/fslcourse/practicals/fdt1/index.html#pipeline

    The steps involved in this pipeline are:
    
    1. convert DICOM files to NIfTI.
    2. locate relevant DTI data and metadata files and saved them separately.
    3. DWI denoising using `dwidenoise` command from MRtrix3 package and 
        TOPUP - Susceptibility-induced distortions corrections. 
        Also, generate acqparams.txt and b0.nii.gz files from existing data 
        to perform this step.
    4. EDDY - Eddy currents corrections. This step will also generate 
        averaged b0 and brain mask NifTI files from existing data to perform
        this step.
    5. DTIFIT - fitting diffusion tensors
"""

import logging
from typing import Union
from pathlib import Path
from dtip.convert import convert_raw_dicom_to_nifti


__all__ = [
    'process_one_subject'
]

def process_one_subject(input_path: Union[str, Path],
                        output_path: Union[str, Path],
                        method: str = "dcm2nii",
                        strip_skull: bool = True,
                        gz: bool = True,
                        reorient: bool = True) -> int:
    """Process DTI data for one subject.

    The steps involved in this pipeline are mentioned at the top.

    Args:
        input_path: path to subject's folder containing DICOM files.
        output_path: folder location where outputs will be saved.
        method: select a DICOM to NIfTI conversion method. Choose one of
            the following conversion methods: `auto` (run both `dcm2nii` and 
            `dcm2niix`), `dcm2nii` (MRICron), and `dcm2niix` (newer version 
            of dcm2nii). 
            [default: `dcm2nii`]
        strip_skull: Whether to remove the skull or not. BET will be used for
            this step with -F flag for 4D data processing
            [default: True]
        gz: compress .nii to .nii.gz. [default: True]
        reorient: reorient the dicoms according to LAS orientation.
            [default: True]

    Returns:
        exit code 0 upon successful execution. The results will be saved
        at `output_path`
    """

    input_path, output_path = Path(input_path), Path(output_path)
    output_path = output_path/input_path.stem
    output_path.mkdir(parents=True, exist_ok=True)
    logging.debug(f"outputs will be saved at `{output_path}`")

    # * Step 1: Convert DICOM data to NIfTI
    logging.debug("Converting DICOM to NIfTI...")
    # Create new folder to store nifti files
    exit_code = convert_raw_dicom_to_nifti(input_path=input_path,
                                           output_path=output_path,
                                           method=method,
                                           gz=gz,
                                           reorient=reorient)
    if exit_code != 0:  # Stop here if any error
        _msg = "Error in `convert_dicom_to_nifti` execution :(. Stopped."
        logging.error(_msg)
        raise RuntimeError(_msg)

    logging.info(f"Saved @ {output_path}")
    logging.info("done.")

    return 0