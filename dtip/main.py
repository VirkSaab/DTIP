import click
import time
import logging
import logging.config
from dtip.utils import load_config, ROOT_DIR
from pyfiglet import Figlet
from typing import Union
from pathlib import Path

# Load project info
CONTEXT_SETTINGS = dict(auto_envvar_prefix="COMPLEX")
# Load Config file
CNF = load_config(ROOT_DIR / "config/default.yaml")


# ----------------------------------------> LOGGING :
logfile_path = f"{CNF.paths.logs_dir}/{CNF.log.filename}.log"
logging.config.fileConfig(
    ROOT_DIR / "config/logs.conf",
    defaults={"logfilename": logfile_path},
    disable_existing_loggers=False,
)


# ----------------------------------------> CLI ::
# * Entry point
@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        # * BANNER
        # Find more fonts here: http://www.figlet.org/examples.html
        f = Figlet(font="smslant")
        banner = " ".join([c for c in CNF["project_name"].upper()])
        # banner = f"..._ {banner} _..."
        click.secho(f"{f.renderText(banner)}", fg="yellow")
        click.echo(
            """Diffusion Tensor Imaging Processing (DTIP)

    This CLI is a Python wrapper heavily based on FSL, DTI-TK, and dcm2nii.
    The purpose of this tool is to automate the processing and
    registering pipeline for multiple subjects. This is not a fully
    automated tool. Manual inspection of data is required to ensure
    the quality of each subject.

    Type `dtip --help` for usage details
    """
        )
        click.echo(ctx.get_help())

    else:
        click.secho(f"\n[@ {ctx.invoked_subcommand}] begin:", fg="cyan")


# ----------------------------------------> CONVERT :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="./converted",
    show_default=True,
    help="folder location to save outputs.",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["auto", "dcm2nii", "dcm2niix"], case_sensitive=False),
    show_default=True,
    default="dcm2nii",
    help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).",
)
@click.option(
    "--multi",
    is_flag=True,
    default=False,
    help="pass --multi for more than one subject.",
)
def convert(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    method: str,
    multi: bool,
):
    """DICOM to NIfTI (.nii or .nii.gz) Conversion."""
    from dtip.convert import convert_raw_dicom_to_nifti

    if multi:
        subjects = [
            p
            for p in input_path.glob("*")
            if (p.is_dir() and (not str(p).startswith(".")))
        ]
        total_subjects = len(subjects)
        print(subjects)
        for i, subject_path in enumerate(subjects, start=1):
            save_folder = output_path / subject_path.stem
            save_folder.mkdir(parents=True, exist_ok=True)
            ret_code = convert_raw_dicom_to_nifti(
                subject_path, save_folder, method=method
            )
            click.echo(f"[{i}/{total_subjects}] extracted at {save_folder}")
    else:
        ret_code = convert_raw_dicom_to_nifti(
            input_path, output_path, method=method)

    if ret_code == 0:
        click.secho("[@ convert] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="./dtitk_converted",
    show_default=True,
    help="folder location to save outputs.",
)
def fsl_dtitk_multi(input_path, output_path):
    """Convert FSL to DTI-TK format for registration for multiple subjects"""
    import os
    from dtip.convert import fsl_to_dtitk_multi

    total_subjects = os.listdir(input_path)
    ret_code = fsl_to_dtitk_multi(input_path, output_path)
    if ret_code == 0:
        click.secho(f"Converted {total_subjects} subjects.", fg="cyan")
        click.secho("[@ fsl-dtitk-multi] completed!\n", fg="green")


# ----------------------------------------> LOCATE :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="./located",
    show_default=True,
    help="folder location to save outputs.",
)
def locate(input_path, output_path):
    """Locate required DTI data files from converted volumes"""
    from dtip.locate import locate_dti_files

    ret_code, _ = locate_dti_files(
        input_path=input_path,
        output_path=output_path,
        protocol_names=CNF.protocol_names,
        n_gradients=CNF.n_gradients,
        ret_paths=False,
    )

    if ret_code == 0:
        click.secho("[@ locate] completed!\n", fg="green")


# ----------------------------------------> GENERATE :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-d",
    "--output_path",
    default="index.txt",
    show_default=True,
    help="path/to/file/index.txt",
)
def make_index(input_path: str, output_path: str):
    """Generate an index.txt file containing value 1 for each DTI volume"""
    from dtip.generate import make_index_file

    ret_code = make_index_file(input_path, output_path)
    if ret_code == 0:
        click.secho("[@ make-index] completed!\n", fg="green")


@cli.command()
@click.option(
    "-t", "--readout_time", default=0.05, show_default=True, help="Total readout time."
)
@click.option(
    "-ap",
    "--ap_pe",
    default="0,-1,0",
    show_default=True,
    help="Anterior to Posterior Phase Encoding.",
)
@click.option(
    "-pa",
    "--pa_pe",
    default="0,1,0",
    show_default=True,
    help="Posterior to Anterior Phase Encoding.",
)
@click.option(
    "-d",
    "--output_path",
    default="acqparams.txt",
    show_default=True,
    help="path/to/file/acqparams.txt",
)
def make_acqparams(readout_time: float, ap_pe: list, pa_pe: list, output_path: str):
    """Generate the acqparams.txt file"""
    from dtip.generate import make_acquisition_params

    ret_code = make_acquisition_params(readout_time, ap_pe, pa_pe, output_path)
    if ret_code == 0:
        click.secho("[@ make-acqparams] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="b0.nii.gz",
    show_default=True,
    help="path/to/file/b0.nii.gz",
)
@click.option("-idx", default=0, show_default=True, help="volume index to extract.")
def make_b0(input_path: str, output_path: str, idx: str):
    """From the DTI 4D data, choose a volume without diffusion weighting 
    (e.g. the first volume). You can now extract this as a standalone 3D image,
    using `fslroi` command. This function runs the `fslroi` command internally.
    """
    from fsl.wrappers.misc import fslroi

    fslroi(input_path, output_path, idx, 1)
    click.secho("[@ make-b0] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("ref_template_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="bootstrapped_template.nii.gz",
    show_default=True,
    help="path/to/save/bootstrapped_template.nii.gz",
)
def bootstrap_template(input_path, ref_template_path, output_path):
    """Create a bootstrap template from the given subjects using DTI's 
        `dti_template_bootstrap` command.
    """
    import os
    import shutil
    import subprocess

    dtitk_dir = f"{ROOT_DIR}/dtitk"
    os.environ["DTITK_ROOT"] = dtitk_dir
    os.environ["PATH"] += f":{dtitk_dir}/bin:{dtitk_dir}/utilities:{dtitk_dir}/scripts"
    subs_filepaths = [
        subject_path / "dti_dtitk.nii.gz"
        for subject_path in Path(input_path).glob("*")
        if subject_path.is_dir()
    ]
    # Create a file with subjects paths
    subs_filepath = Path(output_path).parent / "subs.txt"
    with open(subs_filepath, "w") as subf:
        for filepath in subs_filepaths:
            subf.write(f"{filepath}\n")
    # Run template bootstrap command
    ret_code = subprocess.run(
        ["dti_template_bootstrap", str(ref_template_path), str(subs_filepath)]
    ).returncode
    if ret_code != 0:
        _errmsg = "Something wrong with `dti_template_bootstrap` execution"
        raise RuntimeError(_errmsg)
    # Move the generated mean_initial template to `output_path`
    shutil.move("mean_initial.nii.gz", output_path)
    if ret_code == 0:
        click.secho("[@ bootstrap-template] completed!\n", fg="green")


# ----------------------------------------> PROCESS :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-m", "--mask_path", type=click.Path(exists=True), help="path/to/brain_mask.nii.gz"
)
@click.option(
    "-o",
    "--output_path",
    default="masked.nii.gz",
    show_default=True,
    help="path/to/file/masked.nii.gz",
)
def apply_mask(input_path, mask_path, output_path):
    """Apply mask on the 3D or 4D DTI volume by multiplying the mask with data.
    """
    from fsl.wrappers.fslmaths import fslmaths

    ret_code = fslmaths(input_path).mul(mask_path).run(output_path).returncode
    if ret_code == 0:
        click.secho("[@ apply-mask] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="./processed",
    show_default=True,
    help="folder location to save output files.",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["auto", "dcm2nii", "dcm2niix"], case_sensitive=False),
    show_default=True,
    default="dcm2nii",
    help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).",
)
@click.option(
    "--ss/--no-ss",
    default=True,
    show_default=True,
    help="Perform skull stripping on DTI data. This step will be performed on eddy corrected DTI data.",
)
def process(input_path, output_path, method, ss):
    """Perform DTI processing on one subject.

        INPUT_PATH - path to subject folder or zip file.
    """
    from dtip.process import process_one_subject

    ret_code = process_one_subject(
        input_path=input_path,
        output_path=output_path,
        method=method,
        protocol_names=CNF.protocol_names,
        n_gradients=CNF.n_gradients,
        strip_skull=ss,
    )
    if ret_code == 0:
        click.secho("[@ process] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output_path",
    default="./processed",
    show_default=True,
    help="folder location to save output files.",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["auto", "dcm2nii", "dcm2niix"], case_sensitive=False),
    show_default=True,
    default="dcm2nii",
    help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).",
)
@click.option(
    "-x",
    "--exclude",
    default="",
    type=str,
    show_default=True,
    help="comma separted subject names to exclude.",
)
@click.option(
    "-f",
    "--fracintensity",
    default=0.5,
    type=float,
    show_default=True,
    help="`-f` flag value for FSL's BET command.",
)
@click.option(
    "--ss/--no-ss",
    default=True,
    show_default=True,
    help="Perform skull stripping on DTI data. This step will be performed on eddy corrected DTI data.",
)
def process_multi(input_path, output_path, method, ss, exclude, fracintensity):
    """Perform DTI processing on multiple subjects.

        INPUT_PATH - path to subject folder or zip file.
    """
    from dtip.process import process_multi_subjects

    exclude_list = [v.strip().lstrip().rstrip()
                    for v in str(exclude).split(",")]
    ret_code = process_multi_subjects(
        input_path=input_path,
        output_path=output_path,
        protocol_names=CNF.protocol_names,
        n_gradients=CNF.n_gradients,
        method=method,
        exclude_list=exclude_list,
        bet_f_thresh=fracintensity,
        strip_skull=ss,
    )
    if ret_code == 0:
        click.secho("[@ process-multi] completed!\n", fg="green")
    else:
        _errmsg = f"[@ process-multi] completed (with {ret_code} error subjects)!"
        click.secho(f"{_errmsg}\n", fg="yellow")


# ----------------------------------------> REGISTER :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-t", 
    "template_path", 
    type=click.Path(exists=True), 
    default=None, 
    show_default=True
)
@click.option(
    "-btp",
    "--bootstrapped_template_path",
    type=click.Path(exists=True),
    default=None,
    show_default=True,
)
@click.option(
    "-o",
    "--output_path",
    default="./register_output",
    show_default=True,
    help="path/to/save/register folder",
)
def register_multi(
    input_path: str,
    template_path: str,
    bootstrapped_template_path: Union[str, None],
    output_path: str,
):
    """Perform image registeration using existing template on the given
        subjects using DTI-TK toolkit.

    Args:
        input_path: subject's preprocessed folder path.
            Perform `dtip process-multi` command to preprocess the subjects
            before registration.
        template_path: Path of the template to use for registration. If
            bootstrapped_template_path is given then this template will be
            ignored.
        bootstrapped_template_path: A manually created mean_initial template
            using subjects data. This file can be created using
            `dtip bootstrap-template` CLI command. If this template is given
            step 1 and 2 (which are subject FSL to DTITK conversion and
            initial template bootstrapping) will be skipped. The process will
            start from affine alignment.
        output_path: location to save the registration output files.

    Returns:
        exit code 0 on successful execution.
    """
    from dtip.register import dtitk_register_multi

    if (template_path is None) and (bootstrapped_template_path is None):
        raise ValueError("At least one template is required.")
    ret_code = dtitk_register_multi(
        input_path=input_path,
        template_path=template_path,
        bootstrapped_template_path=bootstrapped_template_path,
        output_path=output_path,
        n_diffeo_iters=CNF.n_diffeo_iters,
    )
    if ret_code == 0:
        click.secho("[@ process-multi] completed!\n", fg="green")


if __name__ == "__main__":
    cli()
