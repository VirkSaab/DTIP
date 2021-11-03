import os
import click
from pyfiglet import Figlet

# * CLI BANNER
# Find more fonts here: http://www.figlet.org/examples.html
f = Figlet(font="smslant")
click.echo(f"{f.renderText('D T I P')}")
CONTEXT_SETTINGS = dict(auto_envvar_prefix="COMPLEX")
ROOTDIR = os.path.abspath(os.path.dirname(__file__))


@click.command(context_settings=CONTEXT_SETTINGS)
def cli():
    msg = """Diffusion Tensor Imaging Processing (DTIP)

    This CLI is a Python wrapper heavily based on FSL, DTI-TK, and dcm2nii.
    The purpose of this tool is to automate the processing and
    registering pipeline for multiple subjects. This is not a fully
    automated tool. Manual inspection of data is required to ensure
    the quality of each subject.

    Type `dtip --help` for usage details
    """
    click.echo(msg)
