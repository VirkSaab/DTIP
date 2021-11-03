"""This module contains some useful functions which can be used throughout the project.
"""

import time
import click
import subprocess
from ruamel.yaml import YAML, yaml_object
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from pathlib import Path
from typing import Any, Callable, Union, Dict, Tuple
from easydict import EasyDict as edict
from collections import OrderedDict


__all__ = [
    # Configuration handlers
    "load_config", "save_config",

    # Functions
    "get_recent_githash",

    # Classes
    "SpinCursor",

    # Decorators
    "show_exec_time",
]

# ======================= CONFIGURATION HANDLERs ===========================

# * When using the decorator, which takes the YAML() instance as a parameter,
# * the yaml = YAML() line needs to be moved up in the file -- *yaml docs*
yaml = YAML(typ='safe', pure=True)
yaml.default_flow_style = False
yaml.indent(mapping=2, sequence=4, offset=2)
ROOT_DIR = Path(__file__).parent.parent


@yaml_object(yaml)
class JoinPath:
    """Custom tag `!join` loader class to join strings for yaml file."""

    yaml_tag = u'!joinpath'

    def __init__(self, joined_string):
        self.joined_string = joined_string

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag,
                                            u'{.joined_string}'.format(node))

    @classmethod
    def from_yaml(cls, constructor, node):
        seq = constructor.construct_sequence(node)
        fullpath = Path('/'.join([str(i) for i in seq])).resolve()
        if len(str(fullpath.name).split(".")) == 1:  # This is a directory
            fullpath.mkdir(parents=True, exist_ok=True)
            # Create a empty .gitkeep file to keep the empty folder structure
            # in git repo
            if len(list(fullpath.glob("**/*"))) == 0:
                (fullpath/".gitkeep").touch(mode=0o666, exist_ok=True)
        return str(fullpath)


@yaml_object(yaml)
class RootDirSetter:
    """Custom tag `!rootdir` loader class for yaml file."""

    yaml_tag = u'!rootdir'

    def __init__(self, path):
        self.path = path

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag,
                                            u'{.path}'.format(node))

    @classmethod
    def from_yaml(cls, constructor, node):
        return str(ROOT_DIR)


def load_config(path: Union[str, Path], pure: bool = False) -> dict:
    """config.yaml file loader.

    This function converts the config.yaml file to `dict` object.

    Args:
        path: config.yaml filepath
        pure: If True, just load the .yaml without converting to EasyDict and exclude extra info.

    Returns:
        `dict` object containing configuration parameters.

    Example:
        .. code-block:: python

            config = load_config("../config.yaml")
            print(config["project_name"])
    """

    path = str(Path(path).absolute().resolve())
    # * Load config file
    with open(path) as file:
        config = yaml.load(file)

    if pure == False:  # Add extra features
        # Convert dict to easydict
        config = edict(config)
        # Save the config filepath itself for further convenience
        config.original_config_filepath = path
    return config


def save_config(config_dict: Dict, path: Union[str, Path] = None,
                saveas_ordered: bool = True, force_overwrite: bool = False,
                file_extension: str = 'yaml') -> None:
    """save `dict` config parameters to `.yaml` file

    Args:
        config_dict: parameters to save.
        path: path/to/save/auto_config.yaml.
        saveas_ordered: save as collections.OrderedDict on Python 3 and 
            `!!omap` is generated for these types.
        file_extension: default `.yaml`

    Returns: 
        nothing

    Example:
        .. code-block:: python

            config = {"Example": 10}
            save_config(config, "../auto_config.yaml")
    """

    if path is None:  # set default path if not given
        path = config_dict.paths.output_dir
        path = f"{path}/modified_config.{file_extension}"
    else:
        path = str(Path(path).absolute().resolve())
        # Check if the path given is the original config's path
        if (config_dict.original_config_filepath == path) and (not force_overwrite):
            msg = f"""
            Error while saving config file @ {path}.
            Cannot overwrite the original config file
            Choose different save location.
            """
            raise ValueError(msg)

    # converting easydict format to default dict because
    # YAML does not processes EasyDict format well.
    cleaned_dict = {
        k: edict_to_dict_converter(v)
        for k, v in config_dict.items()
    }
    # print("cleaned_dict =", cleaned_dict)

    # Fix order
    if saveas_ordered:
        cleaned_dict = OrderedDict(cleaned_dict)

    # Save the file to given location
    with open(path, 'w') as file:
        yaml.dump(cleaned_dict, file)

    print(f"config saved @ {path}")


def edict_to_dict_converter(x: Union[Dict, Any]) -> Union[Dict, Any]:
    """Recursive function to convert given dictionary's datatype
    from edict to dict.

    Args:
        x (dict or other): nested dictionary

    Returns: x (same x but default dict type)
    """
    if not isinstance(x, dict):
        return x

    # Recursion for nested dicts
    ndict = {}
    for k, v in x.items():
        v1 = edict_to_dict_converter(v)
        ndict[k] = v1
    return ndict


# ============================== CLASSES ==============================
class SpinCursor:
    """A waiting animation when program is being executed. 

        `reference source (stackoverflow) <https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running>`_ 

        Args:
            desc : The loader's description. Defaults to "Loading...".
            end : Final print. Defaults to "Done!".
            cursor_type : Set the animation type. Choose one out of
                'bar', 'spin', or 'django'.
            timeout : Sleep time between prints. Defaults to 0.1.

        Example:
            Using *with* context:

            .. code-block:: python

                with SpinCursor("Running...", end=f"done!!"):
                    subprocess.run(['ls', '-l'])
                    time.sleep(10)

            Using normal code:

            .. code-block:: python

                cursor = SpinCursor("Running...", end=f"done!!")
                cursor.start()
                subprocess.run(['ls', '-l'])
                time.sleep(10)
                cursor.stop()

        Returns: 
            Nothing
        """

    def __init__(self, desc: str = "Loading...", end: str = "Done!", cursor_type: str = "bar", timeout: float = 0.1) -> None:

        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)

        if cursor_type == 'bar':
            self.steps = [
                "[=     ]",
                "[ =    ]",
                "[  =   ]",
                "[   =  ]",
                "[    = ]",
                "[     =]",
                "[    = ]",
                "[   =  ]",
                "[  =   ]",
                "[ =    ]",
            ]
        elif cursor_type == 'spin':
            self.steps = ['|', '/', '-', '\\']
        elif cursor_type == 'django':
            self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        else:
            raise NotImplementedError("choose one [`spin`, `bar`, `django`].")

        self.done = False

    def start(self) -> object:
        """Start the animation. See example above."""
        self._thread.start()
        return self

    def _animate(self) -> None:
        for c in cycle(self.steps):
            if self.done:
                break
            print(click.style(f"\r{self.desc} {c}", fg='yellow'), flush=True, end="")
            time.sleep(self.timeout)

    def __enter__(self) -> None:
        self.start()

    def stop(self) -> None:
        """Stop animation. See example above."""
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(click.style(f"\r{self.end}", fg='green'), flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


# ============================== DECORATORS ==============================
def show_exec_time(func: Callable) -> Any:
    """Display the execution time of a function.

        This decorator is suited to large programs that takes
            more than a second to run.

        Example:

            .. code-block:: python

                @show_exec_time
                def take_a_break(timeout=10):
                    time.sleep(timeout)

                >>> take_a_break()
                >>> >> Completed in 00h:00m:10s <<
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Run the given function
        results = func(*args, **kwargs)

        end_time = time.time()

        hrs = (end_time - start_time) // 3600
        rem = (end_time - start_time) % 3600
        mins = rem // 60
        secs = rem % 60

        hrs = str(round(hrs)).zfill(2)
        mins = str(round(mins)).zfill(2)
        secs = str(round(secs)).zfill(2)

        print(f"\n>> Completed in {hrs}h:{mins}m:{secs}s <<\n")

        return results
    return wrapper


# ============================= FUNCTIONS ==============================
def get_recent_githash():
    """Get the recent commit git hash"""
    proc = subprocess.Popen(["git", "rev-parse", "HEAD"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.read().strip().decode('ascii')
