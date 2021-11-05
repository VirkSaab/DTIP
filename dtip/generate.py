from pathlib import Path
from typing import Union
import nibabel as nib

__all__ = ["make_index_file", "make_acquisition_params"]


def make_index_file(
    input_path: Union[str, Path], output_path: Union[str, Path] = "index.txt"
) -> int:
    """Create an index file with value 1 in each row per DTI volume.

    For example, there are 17 DTI volume (4th dimension) in DTIdata.nii.gz.
    Then, the index file will contain value 1 in 17 rows.

    Args:
        input_path: <DTI data file>.nii.gz file path.
        output_path: name and path of the index file like `save/here/index.txt`

    Returns:
        exit_code 0 if successfully executed.
    """

    img = nib.load(input_path)
    x, y, z, t = img.shape
    with open(output_path, "w") as idx_file:
        for _ in range(t):
            idx_file.write(f"1\n")
    return 0


def make_acquisition_params(
    readout_time: float,
    AP_PE: Union[list, str],
    PA_PE: Union[list, str],
    output_path: Union[str, Path] = "acqp.txt",
) -> int:
    """Create the acquisition parameters file.

    This file contains the information with the PE direction, the sign of the
    AP and PA volumes and some timing information obtained by the acquisition. 
    The first three elements of each line comprise a vector that specifies the 
    direction of the phase encoding. The non-zero number in the second column 
    means that is along the y-direction. A -1 means that k-space was traversed 
    Anterior→Posterior and a 1 that it was traversed Posterior→Anterior. 
    The final column specifies the "total readout time", which is the time 
    (in seconds) between the collection of the centre of the first echo and the 
    centre of the last echo.

    Args:
        readout_time: "total readout time", which is the time (in seconds) between the collection of the centre of the first echo and the centre of the last echo.
        AP_PE: Anterior→Posterior Phase Encoding. Such as [0, -1, 0] for y-direction or `j` vector. or '0,-1,0' as a string.
        PA_PE: Posterior→Anterior Phase Encoding.
        output_path: path/to/file/acqparams.txt

    Returns:
        exit_code 0 if successfully executed.
    """
    if isinstance(AP_PE, str):
        AP_PE = AP_PE.split(",")
    if isinstance(PA_PE, str):
        PA_PE = None if PA_PE == "" else PA_PE.split(",")

    first_line = f"{' '.join(map(str, AP_PE))} {readout_time}\n"
    if PA_PE:
        second_line = f"{' '.join(map(str, AP_PE))} {readout_time}"

    with open(output_path, "w") as acq_file:
        acq_file.write(first_line)
        if PA_PE:
            acq_file.write(second_line)

    return 0
