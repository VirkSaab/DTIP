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


# * Add dtitk tool to PATH
from dtip.utils import ROOT_DIR
if os.getenv('DTITK_ROOT') is None:
    # Add DTI-TK PATH as environment variable
    dtitk_maindir = f"{ROOT_DIR}/dtitk"
    os.environ["DTITK_ROOT"] = dtitk_maindir
    os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"
    

__all__ = [
    'dtitk_register_multi',
    'template_to_subject_space',
    'template_to_subject_space_multi'
]


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

    if bootstrapped_template_path is None:
        # * Step 1: Convert FSL format to DTI-TK format and move files
        # * to `output_path`.
        exit_code = fsl_to_dtitk_multi(input_path, output_path)
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
    if Path('mean_initial.nii.gz').exists():
        shutil.move('mean_initial.nii.gz', output_path/'mean_initial.nii.gz')
    if Path('mask.nii.gz').exists():
        shutil.move('mask.nii.gz', output_path/'mask.nii.gz')
    if Path('template_tr.nii.gz').exists():
        shutil.move('template_tr.nii.gz', output_path/'template_tr.nii.gz')
    if Path('bootstrapped_template.nii.gz').exists():
        shutil.move('bootstrapped_template.nii.gz',
                    output_path/'bootstrapped_template.nii.gz')
    logging.info("Registration complete!")
    return 0


def template_to_subject_space(subject_dir_path: Union[str, Path],
                              template_path: Union[str, Path],
                              transform_type: str) -> int:
    """Transform the template to the subject space.
        This function uses DTI-TK tool to perform transformations.

        This function is based on the following tutorial:
            http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.OptionspostReg

    Args:
        subject_dir_path: Subject folder path containing `subj.nii.gz` and
            `subj.aff` files to perform affine or diffeomorphic transformation. 
            If the `transform_type` argument is `diffeo` then `subj.df.nii.gz`
            file should be present as well.
        template_path: Path to the template or atlas file to transform to 
            the subject space.
        transform_type: Either `affine` for affine transformation or `diffeo`
            for diffeomorphic transformation.

    Returns:
        exit code 0 on successful execution
    """

    subject_dir_path = Path(subject_dir_path)
    if not subject_dir_path.is_dir():
        raise ValueError("`subject_dir_path` must be a folder.")
    subject_path = subject_dir_path/'dti_dtitk.nii.gz'
    aff_path = subject_dir_path/'dti_dtitk.aff'
    inv_aff_path = subject_dir_path/'dti_dtitk_inv.aff'
    if transform_type == 'affine':  # perform affine transformation
        savepath = subject_dir_path / \
            Path(template_path).stem.replace('.nii', '')
        savepath = f"{savepath}_affine_dti_space.nii.gz"
        # Compute the inverse of affine matrix
        ret_code = subprocess.run([
            'affine3Dtool', '-in', aff_path, '-invert', '-out', inv_aff_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `affine3Dtool` subprocess.")

        ret_code = subprocess.run([
            'affineScalarVolume', '-in', template_path, '-trans', inv_aff_path,
            '-target', subject_path, '-interp', '1', '-out', savepath
        ]).returncode

        if ret_code != 0:
            raise RuntimeError(
                "Something wrong with `affineScalarVolume` subprocess.")
    elif transform_type == 'diffeo':  # perform diffeomorphic transformation
        savepath = subject_dir_path / \
            Path(template_path).stem.replace('.nii', '')
        savepath = f"{savepath}_diffeo_dti_space.nii.gz"
        #* Combined displacement field from the native space to the template space 
        diffeo_df_path = subject_dir_path/'dti_dtitk_aff_diffeo.df.nii.gz'
        combined_df_path = subject_dir_path/'dti_dtitk_combined.df.nii.gz'
        ret_code = subprocess.run([
            'dfRightComposeAffine', 
            '-aff', aff_path, 
            '-df', diffeo_df_path, 
            '-out', combined_df_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `dfRightComposeAffine` subprocess.")
        #* Mapping the subject data to the template space 
        subject_combined_path = subject_dir_path/'dti_dtitk_combined.nii.gz'
        ret_code = subprocess.run([
            'deformationSymTensor3DVolume',
            '-in', subject_path,
            '-trans', combined_df_path,
            '-target', template_path, 
            '-out', subject_combined_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `dfRightComposeAffine` subprocess.")
        #* Combined displacement field from the template space to the native space
        ## 1. compute the inverse of the affine transformation with the 
        ## command affine3Dtool
        ret_code = subprocess.run([
            'affine3Dtool', '-in', aff_path, '-invert', '-out', inv_aff_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `affine3Dtool` subprocess.")
        ## 2. compute the inverse of the deformable transformation with the 
        ## command dfToInverse
        diffeo_inv_df_path = subject_dir_path/'dti_dtitk_aff_diffeo.df_inv.nii.gz'
        ret_code = subprocess.run([
            'dfToInverse', '-in', diffeo_df_path, '-out', diffeo_inv_df_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `dfToInverse` subprocess.")
        ## 3. combine the two inverted transformations with the command 
        ## dfLeftComposeAffine (not dfRightComposeAffine)
        combined_inv_df_path = subject_dir_path/'dti_dtitk_combined.df_inv.nii.gz'
        ret_code = subprocess.run([
            'dfLeftComposeAffine', 
            '-df', diffeo_inv_df_path,
            '-aff', inv_aff_path,
            '-out', combined_inv_df_path
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `dfLeftComposeAffine` subprocess.")
        #* Mapping the atlas space data to the subject space 
        ret_code = subprocess.run([
            'deformationScalarVolume',
            '-in', template_path, 
            '-trans', combined_inv_df_path,
            '-target', subject_path,
            '-interp', '1',
            '-out', savepath
        ]).returncode
        if ret_code != 0:
            raise RuntimeError("Something wrong with `deformationScalarVolume` subprocess.")
    else:
        raise NotImplementedError("Unknown `transform_type`. Only `affine` and `diffeo` is supported.")
    return 0


@show_exec_time
def template_to_subject_space_multi(subjects_dir_path: Union[str, Path],
                                    template_path: Union[str, Path],
                                    transform_type: str) -> int:
    """Transform the template to the subject space.
        This function uses DTI-TK tool to perform transformations.

        This function is based on the following tutorial:
            http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.OptionspostReg

    Args:
        subjects_dir_path: Parent folder path containing subject folder 
            containing `subj.nii.gz` and `subj.aff` files to perform 
            affine or diffeomorphic transformation. If the `transform_type` 
            argument is `diffeo` then `subj.df.nii.gz` file should be present
            as well.
        template_path: Path to the template or atlas file to transform to 
            the subject space.
        transform_type: Either `affine` for affine transformation or `diffeo`
            for diffeomorphic transformation.

    Returns:
        exit code 0 on successful execution
    """
    error_list = []
    template_path = Path(template_path)
    subjects_paths = [
        p for p in Path(subjects_dir_path).glob('*')
        if p.is_dir()
    ]
    total_subjects = len(subjects_paths)
    logging.debug("Starting template to subject space transform...")
    for i, subject_path in enumerate(subjects_paths, start=1):
        ret_code = template_to_subject_space(subject_dir_path=subject_path,
                                             template_path=template_path,
                                             transform_type=transform_type)
        if ret_code == 0:
            logging.info(
                f"[{i}/{total_subjects}] Transformed `{template_path.name}` to `{subject_path.name}`")
        else:
            logging.warning(
                f"Error transforming {subject_path.stem}. Skipped.")
            error_list.append(subject_path.stem)

    if error_list:
        print("=" * 10, f"{len(error_list)} Subjects with Errors", "=" * 10)
        for es in error_list:
            print(es)
        print("=" * 44)
        return len(error_list)
    else:
        print("=" * 30)
        print("All subjects completed without errors.")
        return 0
