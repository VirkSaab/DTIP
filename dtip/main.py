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
CNF = load_config(ROOT_DIR/"config/default.yaml")


# ----------------------------------------> LOGGING :
logfile_path = f"{CNF.paths.logs_dir}/{CNF.log.filename}.log"
logging.config.fileConfig(ROOT_DIR/"config/logs.conf",
                          defaults={'logfilename': logfile_path},
                          disable_existing_loggers=False)

# ----------------------------------------> CLI ::


# * Entry point
@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        # * BANNER
        # Find more fonts here: http://www.figlet.org/examples.html
        f = Figlet(font="smslant")
        banner = ' '.join([c for c in CNF["project_name"].upper()])
        # banner = f"..._ {banner} _..."
        click.secho(f"{f.renderText(banner)}", fg='yellow')
        click.echo("""Diffusion Tensor Imaging Processing (DTIP)

    This CLI is a Python wrapper heavily based on FSL, DTI-TK, and dcm2nii.
    The purpose of this tool is to automate the processing and
    registering pipeline for multiple subjects. This is not a fully
    automated tool. Manual inspection of data is required to ensure
    the quality of each subject.

    Type `dtip --help` for usage details
    """)
        click.echo(ctx.get_help())

    else:
        click.secho(f"\n[@ {ctx.invoked_subcommand}] begin:", fg='cyan')


# ----------------------------------------> CONVERT :
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', default="./converted", show_default=True, help="folder location to save outputs.")
@click.option('-m', '--method', type=click.Choice(['auto', 'dcm2nii', 'dcm2niix'], case_sensitive=False), show_default=True, default="dcm2nii", help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).")
@click.option('--multi', is_flag=True, default=False, help="pass --multi for more than one subject.")
def convert(input_path: Union[str, Path], output_path: Union[str, Path], method: str, multi: bool):
    """DICOM to NIfTI (.nii or .nii.gz) Conversion."""
    from dtip.convert import convert_raw_dicom_to_nifti

    if multi:
        subjects = [
            p for p in input_path.glob('*')
            if (p.is_dir() and (not str(p).startswith('.')))
        ]
        total_subjects = len(subjects)
        print(subjects)
        for i, subject_path in enumerate(subjects, start=1):
            save_folder = output_path/subject_path.stem
            save_folder.mkdir(parents=True, exist_ok=True)
            ret_code = convert_raw_dicom_to_nifti(
                subject_path, save_folder, method=method)
            click.echo(f"[{i}/{total_subjects}] extracted at {save_folder}")
    else:
        ret_code = convert_raw_dicom_to_nifti(
            input_path, output_path, method=method)

    if ret_code == 0:
        click.secho('[@ convert] completed!\n', fg='green')


# ----------------------------------------> LOCATE :
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', default="./located", show_default=True, help="folder location to save outputs.")
def locate(input_path, output_path):
    """Locate required DTI data files from converted volumes"""
    from dtip.locate import locate_dti_files

    ret_code, _ = locate_dti_files(input_path=input_path,
                                   output_path=output_path,
                                   protocol_names=CNF.protocol_names,
                                   n_gradients=CNF.n_gradients,
                                   ret_paths=False)

    if ret_code == 0:
        click.secho('[@ locate] completed!\n', fg='green')


# ----------------------------------------> GENERATE :
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-d', '--output_path', default='index.txt', show_default=True, help="path/to/file/index.txt")
def make_index(input_path: str, output_path: str):
    """Generate an index.txt file containing value 1 for each DTI volume"""
    from dtip.generate import make_index_file
    ret_code = make_index_file(input_path, output_path)
    if ret_code == 0:
        click.secho('[@ make-index] completed!\n', fg='green')


@cli.command()
@click.option('-t', '--readout_time', default=0.05, show_default=True, help="Total readout time.")
@click.option('-ap', '--ap_pe', default="0,-1,0", show_default=True, help="Anterior to Posterior Phase Encoding.")
@click.option('-pa', '--pa_pe', default="0,1,0", show_default=True, help="Posterior to Anterior Phase Encoding.")
@click.option('-d', '--output_path', default='acqparams.txt', show_default=True, help="path/to/file/acqparams.txt")
def make_acqparams(readout_time: float, ap_pe: list, pa_pe: list, output_path: str):
    """Generate the acqparams.txt file"""
    from dtip.generate import make_acquisition_params
    ret_code = make_acquisition_params(readout_time, ap_pe, pa_pe, output_path)
    if ret_code == 0:
        click.secho('[@ make-acqparams] completed!\n', fg='green')


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', default='b0.nii.gz', show_default=True, help="path/to/file/b0.nii.gz")
@click.option('-idx', default=0, show_default=True, help="volume index to extract.")
def make_b0(input_path: str, output_path: str, idx: str):
    """From the DTI 4D data, choose a volume without diffusion weighting 
    (e.g. the first volume). You can now extract this as a standalone 3D image,
    using `fslroi` command. This function runs the `fslroi` command internally.
    """
    from fsl.wrappers.misc import fslroi
    fslroi(input_path, output_path, idx, 1)
    click.secho('[@ make-b0] completed!\n', fg='green')


# ----------------------------------------> PROCESS :
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-m', '--mask_path', type=click.Path(exists=True), help="path/to/brain_mask.nii.gz")
@click.option('-o', '--output_path', default='masked.nii.gz', show_default=True, help="path/to/file/masked.nii.gz")
def apply_mask(input_path, mask_path, output_path):
    from fsl.wrappers.fslmaths import fslmaths
    ret_code = (fslmaths(input_path)
                    .mul(mask_path)
                    .run(output_path).returncode)
    if ret_code == 0:
        click.secho('[@ apply-mask] completed!\n', fg='green')


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option("-o", "--output_path", default="./processed", show_default=True, help="folder location to save output files.")
@click.option('-m', '--method', type=click.Choice(['auto', 'dcm2nii', 'dcm2niix'], case_sensitive=False), show_default=True, default="dcm2nii", help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).")
@click.option('--ss/--no-ss', default=True, show_default=True, help="Perform skull stripping on DTI data. This step will be performed on eddy corrected DTI data.")
def process(input_path, output_path, method, ss):
    """Perform DTI processing on one subject.

        INPUT_PATH - path to subject folder or zip file.
    """
    from dtip.process import process_one_subject

    ret_code = process_one_subject(input_path=input_path,
                                   output_path=output_path,
                                   method=method,
                                   protocol_names=CNF.protocol_names,
                                   n_gradients=CNF.n_gradients,
                                   strip_skull=ss)
    if ret_code == 0:
        click.secho('[@ process] completed!\n', fg='green')


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option("-o", "--output_path", default="./processed", show_default=True, help="folder location to save output files.")
@click.option('-m', '--method', type=click.Choice(['auto', 'dcm2nii', 'dcm2niix'], case_sensitive=False), show_default=True, default="dcm2nii", help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).")
@click.option("-x", "--exclude", default='', type=str, show_default=True, help="comma separted subject names to exclude.")
@click.option("-f", "--fracintensity", default=0.5, type=float, show_default=True, help="`-f` flag value for FSL's BET command.")
@click.option('--ss/--no-ss', default=True, show_default=True, help="Perform skull stripping on DTI data. This step will be performed on eddy corrected DTI data.")
def process_multi(input_path, output_path, method, ss, exclude, fracintensity):
    """Perform DTI processing on multiple subjects.

        INPUT_PATH - path to subject folder or zip file.
    """
    from dtip.process import process_multi_subjects
    exclude_list = [
        v.strip().lstrip().rstrip()
        for v in str(exclude).split(',')
    ]
    ret_code = process_multi_subjects(input_path=input_path,
                                      output_path=output_path,
                                      protocol_names=CNF.protocol_names,
                                      n_gradients=CNF.n_gradients,
                                      method=method,
                                      exclude_list=exclude_list,
                                      bet_f_thresh=fracintensity,
                                      strip_skull=ss)
    if ret_code == 0:
        click.secho('[@ process-multi] completed!\n', fg='green')
    else:
        _errmsg = f"[@ process-multi] completed (with {ret_code} error subjects)!"
        click.secho(f'{_errmsg}\n', fg='yellow')


if __name__ == '__main__':
    cli()
