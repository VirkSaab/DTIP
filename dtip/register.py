"""This scripts includes DTI registration pipeline using DTI-TK tool. 
I followed the official DTI-TK tutorials. 
The steps listed below are taken from here:
http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Registration
"""

import os
import logging
from typing import Union
from pathlib import Path
from dtip.convert import fsl_to_dtitk_multi
from dtip.utils import show_exec_time


@show_exec_time
def dtitik_register_multi(input_path: Union[str, Path],
                          template_path: Union[str, Path],
                          mean_initial_template_path: Union[str, Path],
                          output_path: Union[str, Path]) -> int:
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
    template_path = Path(template_path)
    mean_initial_template_path = Path(mean_initial_template_path)

    # * Add dtitk tool to PATH
    dtitk_maindir = f"{Path(__file__).parent}/dtitk"
    os.environ["DTITK_ROOT"] = dtitk_maindir
    os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"

    if mean_initial_template_path is None:
        # * Step 1: Convert FSL format to DTI-TK format and move files
        # * to `output_path`.
        exit_code = fsl_to_dtitk_multi(input_path, output_path)
        if exit_code != 0:  # Stop here if any error
            _msg = "Error in `dtitk_register_multi` execution :(. Stopped."
            logging.error(_msg)
            raise RuntimeError(_msg)
        logging.info(
            f"Converted `{input_path}` to DTI-TK format and saved at `{output_path}`.")

        # * SPATIAL NORMALIZATION AND ATLAS CONSTRUCTION
        # * Step 2. Bootstrapping the initial DTI template from the input DTI volumes
        # Get subjects' DTI file paths
        subs_filepaths = []
        for subject_path in output_path.glob("*"):
            if subject_path.is_dir():
                filename = "dti_dtitk.nii.gz"
                filepath = f"{subject_path}/{filename}"
                subs_filepaths.append(filepath)

        # Create a file with subset names
        subs_filepath = output_path/"subs.txt"
        with open(subs_filepath, "w") as subf:
            for filepath in subs_filepaths:
                subf.write(f"{filepath}\n")
