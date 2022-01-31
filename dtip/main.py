from email.policy import default
import os
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

    total_subjects = len(os.listdir(input_path))
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

    if os.getenv('DTITK_ROOT') is None:
        # Add DTI-TK PATH as environment variable
        dtitk_maindir = f"{ROOT_DIR}/dtitk"
        os.environ["DTITK_ROOT"] = dtitk_maindir
        os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"

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
        click.secho(f"template renamed and saved at {output_path}", fg="green")
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
        bet_f_thresh=CNF.frac_intensity,
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
    "--ss/--no-ss",
    default=True,
    show_default=True,
    help="Perform skull stripping on DTI data. This step will be performed on eddy corrected DTI data.",
)
def process_multi(input_path, output_path, method, ss, exclude):
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
        bet_f_thresh=CNF.frac_intensity,
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


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("template_path", type=click.Path(exists=True))
@click.option(
    "-tt",
    "--transform_type",
    type=click.Choice(["affine", "diffeo"], case_sensitive=False),
    show_default=True,
    default="affine",
    help="Use affine or diffeomorphic transformation.",
)
def template_to_subject(input_path, template_path, transform_type):
    """Transform the template to the subject space."""

    from dtip.register import template_to_subject_space

    ret_code = template_to_subject_space(subject_dir_path=input_path,
                                         template_path=template_path,
                                         transform_type=transform_type)
    if ret_code == 0:
        click.secho("[@ template-to-subject] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("template_path", type=click.Path(exists=True))
@click.option(
    "-tt",
    "--transform_type",
    type=click.Choice(["affine", "diffeo"], case_sensitive=False),
    show_default=True,
    default="affine",
    help="Use affine or diffeomorphic transformation.",
)
def template_to_subject_multi(input_path, template_path, transform_type):
    """Transform the template to the subject space for multiple subjects."""

    from dtip.register import template_to_subject_space_multi

    ret_code = template_to_subject_space_multi(subjects_dir_path=input_path,
                                               template_path=template_path,
                                               transform_type=transform_type)
    if ret_code == 0:
        click.secho("[@ template-to-subject-multi] completed!\n", fg="green")
    else:
        click.secho(
            f"[@ template-to-subject-multi] completed (with {ret_code} error subjects)!\n", fg="yellow")


# ----------------------------------------> ANALYSIS :
@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-tn",
    "subject_space_template_name",
    type=str,
    default=None,
    show_default=True,
    help="Name of the subject space transformed pracellation template.\
    This template should be present in the same folder as input_path\
    For example, JHU_128_pcl_dti_space.nii.gz"
)
@click.option(
    "-o",
    "--output_path",
    default="./analysis_output",
    show_default=True,
    help="folder/path/to/save/output files",
)
def compute_stats(input_path, subject_space_template_name, output_path):
    """Compute ROI stats for a subject."""

    if os.getenv('DTITK_ROOT') is None:
        # Add DTI-TK PATH as environment variable
        dtitk_maindir = f"{ROOT_DIR}/dtitk"
        os.environ["DTITK_ROOT"] = dtitk_maindir
        os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"

    if subject_space_template_name is None:
        raise ValueError("Name of the subject space template is required.")

    from dtip.analysis import ComputeSubjectROIStats

    template_path = Path(input_path).parent/subject_space_template_name
    sstats = ComputeSubjectROIStats(input_path=input_path,
                                    subject_space_pcl_path=template_path,
                                    output_path=output_path)
    ret_code = sstats.run()
    if ret_code == 0:
        click.secho("[@ compute-stats] completed!\n", fg="green")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-sn",
    "subject_name",
    type=str,
    default='dti_dtitk.nii.gz',
    show_default=True,
    help="Name of the subject's DTI data nifti file. For example, dti_dtitk.nii.gz"
)
@click.option(
    "-tn",
    "subject_space_template_name",
    type=str,
    default=None,
    show_default=True,
    help="Name of the subject space transformed pracellation template.\
    This template should be present in the same folder as input_path"
)
@click.option(
    "-o",
    "--output_path",
    default="./analysis_output",
    show_default=True,
    help="folder/path/to/save/output files",
)
def compute_stats_multi(input_path,
                        subject_name,
                        subject_space_template_name,
                        output_path):
    """Compute ROI stats for multiple subjects."""
    if os.getenv('DTITK_ROOT') is None:
        # Add DTI-TK PATH as environment variable
        dtitk_maindir = f"{ROOT_DIR}/dtitk"
        os.environ["DTITK_ROOT"] = dtitk_maindir
        os.environ["PATH"] += f":{dtitk_maindir}/bin:{dtitk_maindir}/utilities:{dtitk_maindir}/scripts"

    if subject_name is None:
        raise ValueError("subject name for DTI data nifti file is required.")
    if subject_space_template_name is None:
        raise ValueError("Name of the subject space template is required.")

    from dtip.analysis import ComputeSubjectROIStats

    ret_list, error_list = [], []
    for subject_path in Path(input_path).glob('*'):
        if not subject_path.is_dir():
            continue
        logging.info(f"Computing ROI stats for {subject_path}...")
        template_path = subject_path/subject_space_template_name
        input_path = subject_path/subject_name
        sstats = ComputeSubjectROIStats(input_path=input_path,
                                        subject_space_pcl_path=template_path,
                                        output_path=output_path)
        ret_code = sstats.run()
        if ret_code == 0:
            ret_list.append(ret_code)
            logging.info("done!")
        else:
            ret_list.append(1)
            error_list.append(subject_path)
            logging.error(f"Error in computing stats for {subject_path}")

    if error_list:
        print('='*20, "Subjects with Errors:")
        for err_sub in error_list:
            print(f'\t{err_sub}')
        print('='*44)

    if sum(ret_list) == 0:
        click.secho("[@ compute-stats] completed!\n", fg="green")
    else:
        click.secho(
            f"[@ compute-stats] completed (with {len(error_list)} error subjects)!\n", fg="yellow")


@cli.command()
@click.argument("pre_subjects_filepath", type=click.Path(exists=True))
@click.argument("post_subjects_filepath", type=click.Path(exists=True))
@click.option(
    "-t",
    "test_type",
    type=click.Choice(["pairedttest", "onesamttest"], case_sensitive=False),
    show_default=True,
    default="pairedttest",
    help="Select the type of the test from the given choices.",
)
@click.option(
    "-roi",
    default=None,
    show_default=True,
    help=f"ROI number from 1 to {CNF.n_rois} | `all` | path/to/file.txt.",
)
@click.option(
    "-roigrps",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help=f"ROI group(s) containing multiple selected ROI IDs.",
)
@click.option(
    "-alpha",
    default=0.05,
    show_default=True,
    help=f"Significance value.",
)
@click.option(
    "-save_as",
    default='results.csv',
    show_default=True,
    help=f"Enter .csv results filename [default is `results.csv`].",
)
@click.option('--allvars', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
def test(pre_subjects_filepath: str,
         post_subjects_filepath: str,
         test_type: str,
         roi: Union[str, int],
         roigrps: str,
         alpha: float,
         save_as: str,
         allvars: bool = False,
         verbose: bool = False):

    from dtip.analysis import DataLoader
    import pandas as pd

    full_verbose = verbose

    # at least one ROI is required to proceed
    if (roi is None) and (roigrps is None):
        _errmsg = "At least one ROI is required."
        _errmsg += " Either pass -roi or -roigrps."
        raise AttributeError(_errmsg)
    if (roi is not None) and (roigrps is not None):
        _errmsg = "Cannot handle both -roi and -roigrps at a time."
        _errmsg += " Either pass -roi or -roigrps."
        raise AttributeError(_errmsg)

    # if roi is a path to a file, load data
    if (roi is not None) and roi.endswith('.txt'):
        with open(roi) as roifile:
            roi = list(map(int, roifile.read().strip('\n').split(',')))

    # If ROI groups filepath is given, load data
    roi_groups = None
    if (roigrps is not None):
        if (roigrps[-4:] != '.txt'):
            raise AttributeError("Only .txt file is supported.")

        with open(roigrps, 'r') as grps_file:
            roi_groups = {}
            for grp_num, line in enumerate(grps_file.read().split('\n')):
                roi_groups[grp_num] = []
                for num in line.split(","):
                    roi_groups[grp_num].append(int(num.strip()))

    with open(pre_subjects_filepath) as pref:
        pre_list = pref.read().split('\n')
    with open(post_subjects_filepath) as postf:
        post_list = postf.read().split('\n')

    print("Hypothesis:\n\tH0: Pre and Post subjects are the same.")
    print("\tH1: Pre and Post subjects are not the same.")

    data = DataLoader(pre_list, post_list, n_rois=CNF.n_rois)

    show_common = False  # Display bool flag for common ROIs
    if test_type == 'pairedttest':
        rejected, not_rejected = {}, {}
        for colname in ['fa_mean', 'rd_mean', 'ad_mean']:
            rejected[colname], not_rejected[colname] = [], []
            print()
            print('='*10, colname, '='*10)
            ret_dict = data.paired_t_test(
                colname,
                roi_num=roi,
                roi_groups=roi_groups,
                alpha=alpha
            )
            if roi is not None:
                if isinstance(roi, (int, str, list)):
                    for roi_num, stats in ret_dict.items():
                        t_stat = stats['t_stat']
                        p_value = stats['two_sided_p_value']
                        one_tailed_p_value = round(stats['p_value'], 5)
                        if len(ret_dict) < 3:
                            print(f"ROI {roi_num}:")
                            print(f"\tt_stat = {round(t_stat, 5)}")
                            print(f"\tp_value = {round(p_value, 5)}")
                            print(
                                f"\tone tailed p_value = {one_tailed_p_value}")
                        if one_tailed_p_value <= alpha:
                            if full_verbose:
                                print(
                                    f"\tp_value ({one_tailed_p_value}) < alpha ({alpha}).")
                                print("\t=> null hypothesis H0 rejected.")
                            rejected[colname].append(roi_num)
                            ret_dict[roi_num]['H0_rejected'] = True
                        else:
                            if full_verbose:
                                print(
                                    f"\tp_value ({one_tailed_p_value}) > alpha ({alpha}).")
                                print("\t=> null hypothesis H0 NOT rejected.")
                            not_rejected[colname].append(roi_num)
                            ret_dict[roi_num]['H0_rejected'] = False
                    print(f"H0 rejected = {len(rejected[colname])} times")
                    print(
                        f"H0 not rejected = {len(not_rejected[colname])} times")

                    if len(ret_dict) > 1:
                        show_common = True
                else:
                    for i in range(1, CNF.n_rois + 1):
                        one_tailed_p_value = round(ret_dict[i]['p_value'], 5)
                        if one_tailed_p_value <= alpha:
                            if full_verbose:
                                print(
                                    f"p_value ({one_tailed_p_value}) < alpha ({alpha}).")
                                print("=> null hypothesis H0 rejected.")
                            rejected[colname].append(i)
                            ret_dict[i]['H0_rejected'] = True
                        else:
                            if full_verbose:
                                print(
                                    f"p_value ({one_tailed_p_value}) > alpha ({alpha}).")
                                print("=> null hypothesis H0 NOT rejected.")
                            not_rejected[colname].append(i)
                            ret_dict[i]['H0_rejected'] = False
                    print(f"H0 rejected = {len(rejected[colname])} times")
                    print(
                        f"H0 not rejected = {len(not_rejected[colname])} times")
                    show_common = True

            elif roi_groups is not None:
                for i in roi_groups:
                    group_p_value = round(ret_dict[i]['p_value'], 5)
                    if group_p_value <= alpha:
                        # print(f"p_value ({group_p_value}) < alpha ({alpha}).")
                        # print("=> null hypothesis H0 rejected.")
                        rejected[colname].append(i)
                        ret_dict[i]['H0_rejected'] = True
                    else:
                        # print(f"p_value ({group_p_value}) > alpha ({alpha}).")
                        # print("=> null hypothesis H0 NOT rejected.")
                        not_rejected[colname].append(i)
                        ret_dict[i]['H0_rejected'] = False
                    print(f"group {i} p-value = {group_p_value}")
                print(f"H0 rejected = {len(rejected[colname])} times")
                print(f"H0 not rejected = {len(not_rejected[colname])} times")
                show_common = True
            else:
                raise NotImplementedError("Only support `roi` and `roigrps`")

        # Common rejected ROIs in all variables
        if show_common:
            print('\nCommon rejected ROIs in all variables:')
            for i in range(1, CNF.n_rois + 1):
                is_common = all([
                    True if i in _rois else False
                    for col, _rois in rejected.items()
                ])
                if is_common:
                    print(f"\t* ROI number {i} is rejected in all variables.")
            print('\nCommon NOT rejected ROIs in all variables:')
            for i in range(1, CNF.n_rois + 1):
                is_common = all([
                    True if i in _rois else False
                    for col, _rois in not_rejected.items()
                ])
                if is_common:
                    print(
                        f"\t* ROI number {i} is NOT rejected in all variables.")

        print()  # Newline at the end
        df = pd.DataFrame.from_dict(ret_dict)
        print(df)
        df.to_csv(f"{save_as}.csv")


@cli.command()
@click.argument("pre_subjects_filepath", type=click.Path(exists=True))
@click.argument("post_subjects_filepath", type=click.Path(exists=True))
@click.option(
    "-roi",
    default=None,
    show_default=True,
    help=f"ROI number from 1 to {CNF.n_rois} | `all` | path/to/file.txt.",
)
@click.option(
    "-roig",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help=f"ROI group(s) containing multiple selected ROI IDs per line separated by comma.",
)
@click.option(
    "-alpha",
    default=0.05,
    show_default=True,
    help=f"Significance value.",
)
@click.option(
    "-n_rois",
    type=int,
    default=CNF.n_rois,
    show_default=True,
    help=f"Total number of ROIs.",
)
@click.option(
    "-vars",
    default='fa_mean,rd_mean,ad_mean',
    show_default=True,
    help=f"Variables (separated by comma) to use from the subject .csv files.",
)
@click.option('--combine', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
@click.option(
    "-save_as",
    default='results',
    show_default=True,
    help=f"Enter results filename. [default is `results`].",
)
def paired_ttest(pre_subjects_filepath: str,
                 post_subjects_filepath: str,
                 roi: str,
                 roig: str,
                 alpha: float,
                 n_rois: int,
                 save_as: str,
                 vars: bool = False,
                 combine: bool = False,
                 verbose: bool = False):

    import pandas as pd
    from dtip.analysis import PairedTTest

    print("""Hypothesis:
        H0: Pre and Post subjects are the same.
        H1: Pre and Post subjects are not the same.
    """)

    # CHECKS
    # at least one ROI is required to proceed
    if (roi is None) and (roig is None):
        _errmsg = "At least one ROI is required."
        _errmsg += " Either pass -roi or -roig."
        raise AttributeError(_errmsg)

    if (roi is not None) and (roig is not None):
        _errmsg = "Cannot handle both -roi and -roigrps at the same time."
        _errmsg += " Either pass -roi or -roig."
        raise AttributeError(_errmsg)

    with open(pre_subjects_filepath) as pref:
        pre_list = pref.read().split('\n')

    with open(post_subjects_filepath) as postf:
        post_list = postf.read().split('\n')

    test = PairedTTest(
        pre=pre_list, post=post_list, n_rois=n_rois, fill_missing_roi=True
    )
    var_names = vars.strip(' ').split(',')

    def display_hypothesis(results: dict, verbose: bool = False) -> bool:
        if results['p_value'] <= alpha:
            if verbose:
                print(f"\tp_value ({results['p_value']}) < alpha ({alpha}).")
                print("\t=> null hypothesis H0 rejected.")
            return True  # rejected
        else:
            if verbose:
                print(f"\tp_value ({results['p_value']}) > alpha ({alpha}).")
                print("\t=> null hypothesis H0 NOT rejected.")
            return False

    # if roi is a path to a file, load data
    if roi is not None:
        if roi.endswith('.txt'):
            with open(roi) as roifile:
                roi = list(map(int, roifile.read().strip('\n').split(',')))

        results = test.run_roi(
            var_names=var_names,
            roi_num=roi,
            combine=combine
        )
        if isinstance(roi, str) and roi.isnumeric():
            print(f"ROI {roi}:")
            display_hypothesis(results, verbose=True)
        elif roi == 'all':
            rejected, not_rejected = 0, 0
            for roi_num, stats in results.items():
                if verbose:
                    print(f"ROI {roi_num}:")
                if display_hypothesis(stats, verbose=verbose):
                    rejected += 1
                    results[roi_num]['H0_rejected'] = True
                else:
                    not_rejected += 1
                    results[roi_num]['H0_rejected'] = False
        elif isinstance(roi, list) and combine:
            display_hypothesis(results, verbose=True)
        else:
            rejected, not_rejected = 0, 0
            for roi_num, stats in results.items():
                if verbose:
                    print(f"ROI {roi_num}:")
                if display_hypothesis(stats, verbose=verbose):
                    rejected += 1
                    results[roi_num]['H0_rejected'] = True
                else:
                    not_rejected += 1
                    results[roi_num]['H0_rejected'] = False
            print(f"H0 rejected {rejected}/{len(results)} times.")
            print(f"H0 NOT rejected {not_rejected}/{len(results)} times.")
        df = pd.DataFrame.from_dict(results)
        print(df)
        df.to_csv(f"{save_as}.csv")

    # If ROI groups filepath is given, load data
    elif (roig is not None):
        if (roig[-4:] != '.txt'):
            raise AttributeError("Only .txt file is supported.")

        with open(roig, 'r') as grps_file:
            roig = {}
            for grp_num, line in enumerate(
                grps_file.read().split('\n'), start=1
            ):
                grp_name = f'group {grp_num}'
                roig[grp_name] = []
                for num in line.split(","):
                    roig[grp_name].append(int(num.strip()))
        results = test.run_roi_groups(
            var_names=var_names,
            roi_groups=roig,
        )
        rejected, not_rejected = 0, 0
        for roi_num, stats in results.items():
            print(f'{roi_num.capitalize()}:')
            if display_hypothesis(stats, verbose=True):
                rejected += 1
                results[roi_num]['H0_rejected'] = True
            else:
                not_rejected += 1
                results[roi_num]['H0_rejected'] = False
        print(f"H0 rejected {rejected}/{len(results)} times.")
        print(f"H0 NOT rejected {not_rejected}/{len(results)} times.")
        df = pd.DataFrame.from_dict(results)
        print(df)
        df.to_csv(f"{save_as}.csv")

    else:
        print("Something went wrong :(")


if __name__ == "__main__":
    cli()
