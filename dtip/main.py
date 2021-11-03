import click
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

# ----------------------------------------> CLI BANNER :
# Find more fonts here: http://www.figlet.org/examples.html
f = Figlet(font="smslant")
banner = ' '.join([c for c in CNF["project_name"].upper()])
# banner = f"..._ {banner} _..."
click.secho(f"{f.renderText(banner)}", fg='yellow')

# ----------------------------------------> LOGGING :
logfile_path = f"{CNF.paths.logs_dir}/{CNF.log.filename}"
logging.config.fileConfig(ROOT_DIR/"config/logging.conf",
                          defaults={'logfilename': logfile_path},
                          disable_existing_loggers=False)
# create logger
logger = logging.getLogger(CNF.log.name)

logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')


# ----------------------------------------> CLI ::
# * Entry point
@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Diffusion Tensor Imaging Processing (DTIP)

    This CLI is a Python wrapper heavily based on FSL, DTI-TK, and dcm2nii.
    The purpose of this tool is to automate the processing and
    registering pipeline for multiple subjects. This is not a fully
    automated tool. Manual inspection of data is required to ensure
    the quality of each subject.

    Type `dtip --help` for usage details
    """


# ----------------------------------------> CONVERT :
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', default="./dicom_to_nifti_output", show_default=True, help="folder location to save outputs.")
@click.option('-m', '--method', type=click.Choice(['auto', 'dcm2nii', 'dcm2niix'], case_sensitive=False), show_default=True, default="dcm2nii", help="`auto` (use both dcm2nii and dcm2niix), `dcm2nii` (MRICron), and `dcm2niix` (newer version of dcm2nii).")
@click.option('--multi', is_flag=True, default=False, help="pass --multi for more than one subject.")
def dicom_nifti(input_path: Union[str, Path], output_path: Union[str, Path], method: str, multi: bool):
    """DICOM to NIfTI (.nii or .nii.gz) Conversion."""
    from dtip.convert import convert_raw_dicom_to_nifti

    input_path, output_path = Path(input_path), Path(output_path)
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
            convert_raw_dicom_to_nifti(
                subject_path, save_folder, method=method)
            click.echo(f"[{i}/{total_subjects}] extracted at {save_folder}")
    else:
        convert_raw_dicom_to_nifti(input_path, output_path, method=method)
    _log_msg = f"[info @ `dicom_nifti`] {click.style('done!', fg='green')}"
    click.echo(_log_msg)
    logger.info(_log_msg)


if __name__ == '__main__':
    cli()
