"""This scripts includes DTI registration pipeline using DTI-TK tool. 
I followed the official DTI-TK tutorials. 
The steps listed below are taken from here:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Registration
"""

import os
import shutil
import logging
import subprocess
from typing import Union
from pathlib import Path
from dtip.convert import fsl_to_dtitk_multi
from dtip.utils import show_exec_time


@show_exec_time
def dtitk_register_multi(
    input_path: Union[str, Path],
    template_path: Union[None, str, Path],
    bootstrapped_template_path: Union[None, str, Path],
    output_path: Union[str, Path],
    n_diffeo_iters: int = 8,
) -> int:
    """DTI existing Template-based Image Registration using Diffusion 
        Tensor Imaging ToolKit (DTI-TK)

    Args:
        input_path: folder path containing a subject's data.
        template_path: Path of the template to use for registration.
        output_path: location to save the output files.

    Returns:
        exit code 0 on successful execution.
    """

    input_path, output_path = Path(input_path), Path(output_path)
    if bootstrapped_template_path:
        bootstrapped_template_path = Path(bootstrapped_template_path)

    # * Add dtitk tool to PATH
    # dtitk_maindir = f"{Path(__file__).parent}/dtitk"
    # os.environ["DTITK_ROOT"] = dtitk_maindir
    # os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"

    if bootstrapped_template_path is None:
        # * Step 1: Convert FSL format to DTI-TK format and move files
        # * to `output_path`.
        exit_code = 0  # fsl_to_dtitk_multi(input_path, output_path)
        if exit_code != 0:  # Stop here if any error
            _msg = "Error in `dtitk_register_multi` execution :(. Stopped."
            logging.error(_msg)
            raise RuntimeError(_msg)
        logging.info(
            f"Converted `{input_path}` to DTI-TK format and saved at `{output_path}`."
        )

        # * SPATIAL NORMALIZATION AND ATLAS CONSTRUCTION
        # * Step 2. Bootstrapping the initial DTI template from the input DTI volumes
        # Get subjects' DTI file paths
        subs_filepaths = [
            subject_path / "dti_dtitk.nii.gz"
            for subject_path in output_path.glob("*")
            if subject_path.is_dir()
        ]
        # Create a file with subset names
        subs_filepath = output_path / "subs.txt"
        with open(subs_filepath, "w") as subf:
            for filepath in subs_filepaths:
                subf.write(f"{filepath}\n")

        print(subs_filepath)
        # Run the `dti_template_bootstrap` command
        logging.info("Starting template bootstrapping...")
        subprocess.run(
            [
                "dti_template_bootstrap",
                template_path,
                subs_filepath,
                "EDS",
                "4",
                "4",
                "4",
                "0.0001",
            ]
        )
        logging.info("Bootstrapping complete.")
    else:

        subs_filepaths = [
            subject_path / "dti_dtitk.nii.gz"
            for subject_path in output_path.glob("*")
            if subject_path.is_dir()
        ]
        # Create a file with subset names
        subs_filepath = output_path / "subs.txt"
        with open(subs_filepath, "w") as subf:
            for filepath in subs_filepaths:
                subf.write(f"{filepath}\n")

        template_path = bootstrapped_template_path

    # * Step 3: Rigid Alignment of DTI Volumes
    # As mentioned in tutorial, rigid alignment is not required when using
    # an existing template.

    # * Step 4: Affine Alignment of DTI volumes
    logging.info("Affine aligment with template refinement")
    for i, subject_path in enumerate(subs_filepaths, start=1):
        logging.info(
            f"[{i}/{len(subs_filepaths)}] Affine alignment of `{subject_path}`..."
        )
        subprocess.run(
            [
                "dti_affine_reg",
                template_path,
                subject_path,
                "EDS",
                "4",
                "4",
                "4",
                "0.001",
            ]
        )
    logging.info("Affine alignment completed!")

    # * Step 5: Deformable alignment with template refinement
    # generate the mask image for mask argument of diffeomorphic command
    logging.debug("Extract TR image from template to create a mask...")
    ret_code = subprocess.run(
        ["TVtool", "-in", template_path, "-tr", "-out", "template_tr.nii.gz"]
    ).returncode
    if ret_code != 0:
        raise RuntimeError(
            """[Error @ `dtitk_register_multi`]
        Problem with mask generation for diffeomorphic alignment."""
        )
    logging.debug("Generating mask for diffeomorphic alignment...")
    ret_code = subprocess.run(
        [
            "BinaryThresholdImageFilter",
            "template_tr.nii.gz",
            "mask.nii.gz",
            "0.01",
            "100",
            "1",
            "0",
        ]
    ).returncode
    if ret_code != 0:
        raise RuntimeError(
            """[Error @ `dtitk_register_multi`]
        Problem with BinaryThresholdImageFilter command in subprocess."""
        )
    logging.info("Done!")
    # Run alignment subject wise
    # Get subjects' DTI file paths
    for subject_path in output_path.glob("*"):
        if subject_path.is_dir():
            logging.info(
                f"Diffeomorphic alignment of subject `{subject_path}`...")
            filepath = f"{subject_path}/dti_dtitk_aff.nii.gz"
            # Run alignment
            subprocess.run(
                [
                    "dti_diffeomorphic_reg",
                    template_path,
                    filepath,
                    "mask.nii.gz",
                    "1",
                    str(n_diffeo_iters),
                    "0.002",
                ]
            )
            logging.info("Subject alignment complete!")

    # Move extra generated files to output_path folder

    logging.info(f"Moving generated files to `{output_path}`...")
    # mean_initial.nii.gz file
    shutil.move('mean_initial.nii.gz', output_path/'mean_initial.nii.gz')
    shutil.move('mask.nii.gz', output_path/'mask.nii.gz')
    shutil.move('template_tr.nii.gz', output_path/'template_tr.nii.gz')
    shutil.move('bootstrapped_template.nii.gz', 
                output_path/'bootstrapped_template.nii.gz')
    logging.info("Registration complete!")
    return 0
