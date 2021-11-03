"""This module implements the DTI preprocessing pipeline for one and multi
    subject DTI data using FSL in majority.

    Source: https://open.win.ox.ac.uk/pages/fslcourse/practicals/fdt1/index.html#pipeline

    The steps involved in this pipeline are:
    
    1. convert DICOM files to NIfTI.
    2. locate relevant DTI data and metadata files and saved them separately.
    3. DWI denoising using `dwidenoise` command from MRtrix3 package and 
        TOPUP - Susceptibility-induced distortions corrections. 
        Also, generate acqparams.txt and b0.nii.gz files from existing data 
        to perform this step.
    4. EDDY - Eddy currents corrections. This step will also generate 
        averaged b0 and brain mask NifTI files from existing data to perform
        this step.
    5. DTIFIT - fitting diffusion tensors
"""

def process_one_subject():
    pass