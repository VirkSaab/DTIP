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
             json_path: Union[str, Path] = None) -> int:
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
            f"--out={output_path}",
        ]
        if topup_path:
            command.append(f"--topup={str(topup_path)}")
        if json_path:
            command.append(f"--json={str(json_path)}")
        # command.append('-v')
        ret = subprocess.run(command).returncode
        if ret != 0:
            return 1
    return 0


def run_dtifit(input_path: Union[str, Path],
               brain_mask_path: Union[str, Path],
               bvecs_path: Union[str, Path],
               bvals_path: Union[str, Path],
               output_path: Union[str, Path]) -> int:
    """Perform tensor fitting for the given DTI data.

    Args:
        input_path: compressed NIfTI (.nii.gz) file path containing 4D DTI 
        data. Usually, after topup and eddy current corrections.
        brain_mask_path: bet brain binary mask file path.
        bvecs_path: path/to/bvectors.bvec.
        bvals_path: path/to/bvalues.bval.
        output_path: path/to/save/folder/output basename. For example, 
            `output_path=data/dtifitted/dti`. The output files will be named 
            as `dti_FA.nii.gz`, `dti_MD.nii.gz`, `dti_V1.nii.gz`, etc.

    Returns:
        returncode of the subprocess.
    """

    with SpinCursor("Running dtifit...", end=f"Tensor fitted data saved at `{output_path}`"):
        ret_code = subprocess.run([
            "dtifit",
            f"--data={input_path}",
            f"--mask={brain_mask_path}",
            f"--bvecs={bvecs_path}",
            f"--bvals={bvals_path}",
            f"--out={output_path}"
        ]).returncode

    return ret_code


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
        bet_f_thresh: `-f` flag value for FSL's BET command.
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
    ret_code = convert_raw_dicom_to_nifti(input_path=input_path,
                                          output_path=nifti_path,
                                          method=method,
                                          gz=gz,
                                          reorient=reorient)
    if ret_code != 0:  # Stop here if any error
        _errmsg = "Error in `convert_dicom_to_nifti` execution :(. Stopped."
        logging.error(_errmsg)
        raise RuntimeError(_errmsg)

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
            info_dict = json.load(jfile)
            ro_time = info_dict['EchoTime']
            # For eddy command
            json_path = dti_paths['json'] if 'SliceTiming' in info_dict else None
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
    logging.info('Running eddy...')
    ret_code = run_eddy(input_path=dti_paths['nifti'],
                        brain_mask_path=brain_mask_path,
                        index_path=index_path,
                        acqp_path=acqp_path,
                        bvecs_path=dti_paths['bvec'],
                        bvals_path=dti_paths['bval'],
                        topup_path=topup_output_path,
                        output_path=eddy_output_path,
                        # Multiband info (Getting `SliceTiming` error)
                        json_path=json_path
                        )
    if ret_code != 0:  # Stop here if any error
        _msg = "Error in `run_eddy` execution :(. Stopped."
        logging.error(_msg)
        raise RuntimeError(_msg)
    logging.info("Eddy completed!")

    eddy_corrected_dti = str(eddy_output_path).replace('.nii.gz', '')
    eddy_corrected_dti_mask = f"{eddy_corrected_dti}_mask"

    # If True, Strip skull of eddy corrected 4D DTI data using BET with -F flag
    if strip_skull:
        logging.info(f"Applying brain mask to remove non-brain parts...",
                     end=' ')
        ret_code = (fslmaths(eddy_output_path)
                    .mul(brain_mask_path)
                    .run(eddy_output_path).returncode)
        logging.info("done.")
        if ret_code != 0:  # Stop here if any error
            _msg = "Error in fslmaths while applying brain mask :(. Stopped."
            logging.error(_msg)
            raise RuntimeError(_msg)
        # generate new eddy mask based on skull stripped data
        ret_code = subprocess.run([
            'fslmaths', eddy_output_path, '-thrP', '10', '-bin', eddy_corrected_dti_mask
        ]).returncode
        if ret_code != 0:  # Stop here if any error
            _msg = "Error in new mask generation :(. Stopped."
            logging.error(_msg)
            raise RuntimeError(_msg)

    # * DTIFIT - fitting diffusion tensors
    processed_path = output_path/f"3_processed/{subject_name}"
    processed_path.mkdir(parents=True, exist_ok=True)
    fit_output_path = processed_path/'dti'
    logging.debug("Running dtifit...")

    ret_code = run_dtifit(eddy_corrected_dti,
                          eddy_corrected_dti_mask,
                          bvecs_path=dti_paths['bvec'],
                          bvals_path=dti_paths['bval'],
                          output_path=fit_output_path)
    if ret_code != 0:  # Stop here if any error
        _msg = "Error in `run_dtifit` execution :(. Stopped."
        logging.error(_msg)
        raise RuntimeError(_msg)
    logging.debug(f"Tensor fitted file saved @ `{fit_output_path}`")
    return 0


def process_multi_subjects(input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           protocol_names: list,
                           n_gradients: int,
                           method: str = "dcm2nii",
                           exclude_list: list = [],
                           bet_f_thresh: float = 0.5,
                           strip_skull: bool = True,
                           gz: bool = True,
                           reorient: bool = True) -> int:
    """Process DTI data for multiple subjects.

    The steps involved in this pipeline are mentioned above.

    Args:
        input_path: path to subjects data where each subject's data can be
            zip file or folder containing DICOM files.
        output_path: folder location where outputs will be saved.
        protocol_names: list of names of DTI protocol names to locate DTI data 
            files from raw data.
        n_gradients: Number of gradients directions in a DTI volume.
        method: select a DICOM to NIfTI conversion method. Choose one of
            the following conversion methods: `auto` (whichever works best for
            each subject), `dicom2nifti` (python package), `dcm2nii` (MRICron),
            and `dcm2niix` (newer version of dcm2nii). 
            [default: `auto`]
        exclude_list: add the name of the subjects you do not want to process
            in the folder.
        bet_f_thresh: `-f` flag value for FSL's BET command.
        strip_skull: Whether to remove the skull or not. BET will be used for
            this step with -F flag for 4D data processing. 
            [default: True]
        gz: compress .nii to .nii.gz. [default: True]
        reorient: reorient the dicoms according to LAS orientation.
            [default: True]

    Returns:
        exit code 0 upon successful execution.
        Otherwise, throws corresponding error
    """

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    subjects_paths, error_list = list(input_path.glob("*")), []
    total_subjects = len(subjects_paths)
    for i, subject_path in enumerate(subjects_paths, start=1):
        if (subject_path.stem in exclude_list) and (len(exclude_list) != 0):
            logging.info(
                f"Subject {subject_path} is in the excluded list. Skipped.")
            continue
        logging.info(
            f"{'='*15}[{i}/{total_subjects}] Processing `{subject_path}`{'='*15}")
        # Run processing steps for each subject
        try:
            exit_code = process_one_subject(input_path=subject_path,
                                            output_path=output_path,
                                            method=method,
                                            protocol_names=protocol_names,
                                            n_gradients=n_gradients,
                                            bet_f_thresh=bet_f_thresh,
                                            strip_skull=strip_skull,
                                            gz=gz,
                                            reorient=reorient)
            if exit_code == 0:
                logging.info(f"Proessing completed for `{subject_path}`.")
            else:
                _msg = f"Error in processing subject `{subject_path}`"
                logging.error(_msg)
                raise RuntimeError(_msg)

        except:
            _msg = f"Error in processing subject `{subject_path}`"
            error_list.append(subject_path)
            logging.error(_msg)

    if error_list:
        print("="*10, f"{len(error_list)} Subjects with Errors", "="*10)
        for es in error_list:
            print(es)
        print("="*40)
        return len(error_list)
    else:
        print("="*30)
        print("All subjects completed without errors.")
        return 0
