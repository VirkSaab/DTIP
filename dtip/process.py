"""This module implements the DTI preprocessing pipeline for one and multi
    subject DTI data using FSL in majority.

    Source: https://open.win.ox.ac.uk/pages/fslcourse/practicals/fdt1/index.html#pipeline

    The steps involved in this pipeline are:
    
    1. DICOM to NIfTI conversion
    2. TOPUP - Susceptibility-induced distortions corrections.
        Also, generate acqparams.txt and b0.nii.gz files from existing data 
        to perform this step.
    3. EDDY - Eddy currents corrections. This step will also generate 
        averaged b0 and brain mask NifTI files from existing data to perform
        this step.
    4. DTIFIT - fitting diffusion tensors
"""

import json
import logging
import subprocess
import nibabel as nib
from typing import Union
from pathlib import Path
from fsl.wrappers.misc import fslroi
from fsl.wrappers import eddy as fsleddy
from fsl.wrappers.fslmaths import fslmaths
from fsl.wrappers.bet import bet
from dtip.utils import SpinCursor
from dtip.convert import convert_raw_dicom_to_nifti
from dtip.locate import locate_dti_files
from dtip.generate import make_acquisition_params, make_index_file


__all__ = [
    'process_one_subject', 'run_topup', 'run_eddy'
]


def run_topup(input_path: Union[str, Path],
              acqp_path: Union[str, Path],
              output_path: Union[str, Path]) -> int:
    """Perform TOPUP distortions correction step for the given 
        nifti (.nii.gz) DTI data.

    Args:
        input_path: compressed NIfTI (.nii.gz) file path.
        acqp_path: acquisition parameters file path.
        output_path: path/to/save/folder/output basename. For example, 
            `output_path=data/corrected/topup_b0`. Then, the output files will 
            `topup_b0_fieldcoef.nii.gz`, `topup_b0_iout.nii.gz`, 
            `topup_b0_fout.nii.gz`, and `topup_b0_movepar.txt`.

    Returns:
        exit code 0 if completed successfully.
    """

    iout_output_path = f"{output_path}_iout"
    fout_output_path = f"{output_path}_fout"

    # Check if both b0 are present
    _b0shape = nib.load(input_path).shape
    # If there are two b0 images i.e. shape=X,Y,Z,2 then perform TOPUP.
    if (len(_b0shape) == 4) and (_b0shape.shape[3] == 2):
        with SpinCursor("Running topup...", end=f"Saved at `{output_path}`"):
            fsleddy.topup(imain=str(input_path),
                          datain=str(acqp_path),
                          out=str(output_path),
                          iout=iout_output_path,
                          fout=fout_output_path,
                          verbose=True)
    else:
        _msg = 'b0 does not match AP-PA requirements. '
        _msg += 'Skipping TOPUP and returning `input_path`'
        logging.info(_msg)
        return input_path
    return 0


def run_eddy(input_path: Union[str, Path],
             brain_mask_path: Union[str, Path],
             index_path: Union[str, Path],
             acqp_path: Union[str, Path],
             bvecs_path: Union[str, Path],
             bvals_path: Union[str, Path],
             topup_path: Union[str, Path] = None,
             output_path: Union[str, Path] = "eddy_unwarped",
             json_path: Union[str, Path] = None,
             flm: str = "linear",
             fwhm: int = 0,
             shelled: bool = False) -> int:
    """Perform eddy currents distortion correction step for the given 
        nifti (.nii.gz) DTI data. Internally runs FSL's eddy command.

    Args:
        input_path: compressed NIfTI (.nii.gz) file path. Usually, after 
            topup corrections.
        brain_mask_path: path/to/binary_brain_mask.nii.gz.
        index_path: path/to/index.txt file.
        acqp_path: path/to/acqparams.txt.
        bvecs_path: path/to/bvectors.bvec.
        bvals_path: path/to/bvalues.bval.
        topup_path: path/to/topup base name. 
            For example, topup_b0<some name>.<.nii.gz or .txt>
        output_path: path/to/save/folder/output basename. 
            For example, `output_path=processed_data/eddy_unwarped_images`.
            Then, the output files will `eddy_unwarped_images.nii.gz`, 
            `eddy_unwarped_images.eddy_parameters`, 
            `eddy_unwarped_images.eddy_command_txt`, etc.
        json_path: Name of .json text file with information about slice timing.
            N.B. --json and --slspec are mutually exclusive.
        flm: First level EC model (movement/linear/quadratic/cubic, default quadratic)
        fwhm: FWHM for conditioning filter when estimating the parameters (default 0)
        shelled: Assume, don't check, that data is shelled (default false)


    Returns:
        exit code 0 if completed successfully.
    """

    with SpinCursor("Running eddy...", end=f"Eddy corrected saved @ `{output_path}`"):
        command = [
            "eddy",
            f"--imain={input_path}",
            f"--mask={brain_mask_path}",
            f"--index={index_path}",
            f"--acqp={acqp_path}",
            f"--bvecs={bvecs_path}",
            f"--bvals={bvals_path}",
            f"--fwhm={fwhm}",
            f"--flm={flm}",
            f"--out={output_path}",
            # Estimate how susceptibility field changes with subject movement
            "--estimate_move_by_susceptibility",
            "--repol"  # Detect and replace outlier slices
            # "--mporder=16" # Not implemented for CPU
        ]
        if topup_path:
            command.append(f"--topup={str(topup_path)}")
        if json_path:
            command.append(f"--json={str(json_path)}")
        if shelled:
            command.append("--data_is_shelled")
        command.append('-v')
        ret = subprocess.run(command).returncode
        if ret != 0:
            return 1
    return 0


def process_one_subject(input_path: Union[str, Path],
                        output_path: Union[str, Path],
                        method: str,
                        protocol_names: list,
                        n_gradients: int,
                        bet_f_thresh: float = 0.5,
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
        protocol_names: list of names of DTI protocol names to locate DTI data 
            files from raw data.
        n_gradients: Number of gradients directions in a DTI volume.
        strip_skull: Whether to remove the skull or not. BET will be used for
            this step with -F flag for 4D data processing
            [default: True]
        gz: compress .nii to .nii.gz. [default: True]
        reorient: reorient the dicoms according to LAS orientation.
            [default: True]

    Returns:
        return code 0 upon successful execution. The results will be saved
        at `output_path`
    """

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    subject_name = input_path.stem
    logging.debug(f"outputs will be saved at `{output_path}`")

    # * Convert DICOM data to NIfTI
    logging.debug("Converting DICOM to NIfTI...")
    # Create new folder to store nifti files
    nifti_path = output_path/f'1_nifti/{subject_name}'
    # ret_code = convert_raw_dicom_to_nifti(input_path=input_path,
    #                                       output_path=nifti_path,
    #                                       method=method,
    #                                       gz=gz,
    #                                       reorient=reorient)
    # if ret_code != 0:  # Stop here if any error
    #     _errmsg = "Error in `convert_dicom_to_nifti` execution :(. Stopped."
    #     logging.error(_errmsg)
    #     raise RuntimeError(_errmsg)

    # * Locate required DTI data files
    interm_path = output_path/f'2_interm/{subject_name}'
    ret_code, dti_paths = locate_dti_files(input_path=nifti_path,
                                           output_path=interm_path,
                                           protocol_names=protocol_names,
                                           n_gradients=n_gradients,
                                           ret_paths=False)
    if ret_code != 0:  # Stop here if any error
        _errmsg = "Error in `locate_dti_files` execution :(. Stopped."
        logging.error(_errmsg)
        raise RuntimeError(_errmsg)

    # * TOPUP - Susceptibility-induced Distortions Corrections.
    # Make b0 image
    b0_path = interm_path/'b0.nii.gz'
    fslroi(str(dti_paths['nifti']), str(b0_path), 0, 1)
    logging.info(f"Saved b0 @ `{b0_path}`")

    # Make acquisition parameters file
    acqp_path = interm_path/"acqparams.txt"
    if 'json' in dti_paths.keys():
        with open(str(dti_paths['json'])) as jfile:
            ro_time = json.load(jfile)['EchoTime']
    else:
        ro_time = 0.05
    ret_code = make_acquisition_params(ro_time, [0, -1, 0], None, acqp_path)
    if ret_code != 0:  # Stop here if any error
        _errmsg = "Error in `make_acquisition_params` execution :(. Stopped."
        logging.error(_errmsg)
        raise RuntimeError(_errmsg)
    logging.info(f"Saved acq params @ `{acqp_path}`")

    # TOPUP
    # means if TOPUP is performed or not
    topup_basename = "topup_b0"
    topup_output_path = interm_path/topup_basename
    topup_i_output_path = interm_path/f'{topup_basename}_iout'
    ret = run_topup(b0_path, acqp_path, topup_output_path)
    # Means TOPUP is skipped and input_path is returned
    if isinstance(ret, (Path, str)):
        topup_i_output_path, topup_output_path = ret, None
    # Stop here if any error
    if isinstance(ret, int) and (ret_code != 0):
        _errmsg = "Error in `run_topup` execution :(. Stopped."
        logging.error(_errmsg)
        raise RuntimeError(_errmsg)

    # * EDDY - Eddy currents corrections.
     # compute the average image of the corrected b0 volumes
    b0_avg_path = str(interm_path/"b0_avg")
    fslmaths(str(topup_i_output_path)).Tmean().run(b0_avg_path)
    logging.info(f"Created avg b0 file at `{b0_avg_path}`.")
    # use BET on the averaged b0. create a binary brain mask, with a fraction
    # intensity threshold of 0.5.
    bet(input=b0_avg_path, output=b0_avg_path,
        mask=True, robust=True, fracintensity=bet_f_thresh)
    brain_mask_path = f"{b0_avg_path}_mask.nii.gz"
    logging.info(f"Created b0 brain mask file at `{brain_mask_path}`.")
    # Create index.txt file
    index_path = interm_path/"index.txt"
    ret_code = make_index_file(dti_paths['nifti'], output_path=index_path)
    if ret_code != 0:  # Stop here if any error
        _msg = "Error in `generate_index` execution :(. Stopped."
        logging.error(_msg)
        raise RuntimeError(_msg)
    logging.info(f"Created index file at `{index_path}`.")
    # Run eddy
    eddy_output_path = interm_path/"eddy_unwarped"
    json_path = dti_paths['json'] if 'json' in dti_paths.keys() else None
    logging.info('Running eddy...')
    print()
    print(dti_paths['nifti'])
    print(brain_mask_path)
    print(index_path)
    print(acqp_path)
    print(dti_paths['bvec'])
    print(dti_paths['bval'])
    print(topup_output_path)
    print(eddy_output_path)
    print()
    ret_code = run_eddy(input_path=dti_paths['nifti'],
                        brain_mask_path=brain_mask_path,
                        index_path=index_path,
                        acqp_path=acqp_path,
                        bvecs_path=dti_paths['bvec'],
                        bvals_path=dti_paths['bval'],
                        topup_path=topup_output_path,
                        output_path=eddy_output_path,
                        json_path=json_path # Multiband info (Getting `SliceTiming` error)
                        )
    if ret_code != 0:  # Stop here if any error
        _msg = "Error in `run_eddy` execution :(. Stopped."
        logging.error(_msg)
        raise RuntimeError(_msg)
    logging.info("Eddy completed!")

    # # If True, Strip skull of eddy corrected 4D DTI data using BET with -F flag
    # if strip_skull:
    #     logging.info(f"Striping skull of DTI eddy corrected data...")
    #     dti_skull_strip(eddy_output_path, eddy_output_path)
    #     logging.info("done.")

    # logging.info(f"Saved @ {output_path}")
    # logging.info("done.")

    return 0
