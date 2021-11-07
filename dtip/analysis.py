import os
import logging
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from typing import Union
from fastprogress import progress_bar
from fsl.wrappers.fslmaths import fslmaths


__all__ = ['ComputeSubjectROIStats', 'compute_mp_fn']


class ComputeSubjectROIStats:
    def __init__(self,
                 input_path: Union[str, Path],
                 subject_space_pcl_path: Union[str, Path],
                 output_path: Union[str, Path],
                 stats_filename: str = 'dti_stats.csv',
                 show_pb: bool = True):
        """Compute ROI stats for one subject.

            Stats include mean and standard deviation of
            fractional anisotropy (FA), axial diffusivity (AD),
            and radial diffusivity (RD).

        Args:
            input_path: subject's DTI volume nifti file path.
            subject_space_pcl_path: Transformed to subject space ROI
                atlas nifti file path.
            output_path: where the output ROI files will be saved.
            stats_filename: name of the csv file containing computed stats.
                This file will be saved in `output_path`.
            show_pb: Enable/disable progress bar visualization. 
                Useful to disable in multiprocessing.
        """
        self.input_path = Path(input_path)
        self.subject_space_pcl_path = Path(subject_space_pcl_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.show_pb = show_pb

        self.subject_name = self.input_path.parent.stem
        # Set save path for csv file
        self.filepath = self.output_path/f"{self.subject_name}__{stats_filename}"
        # Set a temporary file path for computation operations
        self.tmp_path = self.output_path/f"{self.subject_name}_tmp_roi.nii.gz"

    def run(self) -> int:
        # Get number of ROIs in parcellation / roi file.
        pcl_data = nib.load(self.subject_space_pcl_path)
        self.rois_values = [
            int(i)
            for i in np.unique(pcl_data.get_fdata())
            if i != 0  # Zero is for background
        ]
        self.n_rois = len(self.rois_values)
        print(f"Found {self.n_rois} unique ROI values.")

        # Compute ROI stats
        self.stats_df = self.make_subject_rois()
        self.stats_df.to_csv(self.filepath, index=False)

        # Remove temporary files
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

        return 0

    def make_subject_rois(self) -> pd.DataFrame:
        """Split each ROI, binary threshold, and save as
            nifti files for further computation
        """
        input_path = str(self.input_path)
        subject_stats = {
            'roi_num': [],
            'fa_mean': [], 'fa_std': [],
            'ad_mean': [], 'ad_std': [],
            'rd_mean': [], 'rd_std': [],
        }

        for roi_num in progress_bar(self.rois_values, display=self.show_pb):
            print("ROI_NUM =", roi_num)
            # Create binarized single ROI image
            (fslmaths(self.subject_space_pcl_path)
             .thr(roi_num)  # Threshold
             .uthr(roi_num)  # Upper threshold
             .bin()  # Binary thresholding
             .run(self.tmp_path))
            # Apply the ROI on subject
            fslmaths(input_path).mul(self.tmp_path).run(self.tmp_path)

            # Compute ROI stats
            roi_stats_dict = self._compute(self.tmp_path)
            subject_stats['roi_num'].append(roi_num)
            for k, v in roi_stats_dict.items():
                subject_stats[k].append(v)

        # Remove temporary file
        os.remove(self.tmp_path)

        return pd.DataFrame(subject_stats)

    def _compute(self, roi_path) -> dict:
        """Compute stats using given metrics and operations"""
        roi_stats = {}

        # Fractional Anisotropy (FA)
        ret_code = subprocess.run([
            'TVtool', '-in', roi_path, '-fa', '-out', 'tmp_roi.nii.gz'
        ]).returncode
        if ret_code == 0:
            fa_img = nib.load('tmp_roi.nii.gz').get_fdata()
            roi_stats['fa_mean'] = fa_img.mean()
            roi_stats['fa_std'] = fa_img.std()

        # Axial Diffusivity (AD)
        ret_code = subprocess.run([
            'TVtool', '-in', roi_path, '-ad', '-out', 'tmp_roi.nii.gz'
        ]).returncode
        if ret_code == 0:
            ad_img = nib.load('tmp_roi.nii.gz').get_fdata()
            roi_stats['ad_mean'] = ad_img.mean()
            roi_stats['ad_std'] = ad_img.std()

        # Radial Diffusivity (RD)
        ret_code = subprocess.run([
            'TVtool', '-in', roi_path, '-rd', '-out', 'tmp_roi.nii.gz'
        ]).returncode
        if ret_code == 0:
            rd_img = nib.load('tmp_roi.nii.gz').get_fdata()
            roi_stats['rd_mean'] = rd_img.mean()
            roi_stats['rd_std'] = rd_img.std()
        return roi_stats


def compute_mp_fn(kwargs):
    """Compute wrapper function for compute-stats-multi multiprocessing command
    """

    input_path = kwargs['input_path']
    template_path = kwargs['template_path']
    output_path = kwargs['output_path']
    logging.info(f"Computing ROI stats for {input_path}...")
    sstats = ComputeSubjectROIStats(input_path=input_path,
                                    subject_space_pcl_path=template_path,
                                    output_path=output_path,
                                    show_pb=False)
    ret_code = sstats.run()
    if ret_code == 0:
        logging.info("done!")
    else:
        logging.error(f"Error in computing stats for {input_path}")
    return 0
