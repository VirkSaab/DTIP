import os
import logging
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from typing import Tuple, Union, List, Dict
from fastprogress import progress_bar


__all__ = ['compute_mp_fn', 'ComputeSubjectROIStats', 'PairedTTest']


# ============================== FUNCTIONS ==============================
def fill_missing_roi_values(df: pd.DataFrame, n_rois: int) -> pd.DataFrame:
    """Fill missing values with mean of that column.
    This function will check `roi_num` column from 1 to `n_rois` values,
    if there is any missing number then it will be filled using the mean
    value of the column.

    Args:
        df: pandas dataframe containing data
        n_rois: number of ROI's in the dataframe.
    """
    if df.roi_num.unique().shape[0] != n_rois:
        missing_rois = list(set(range(1, n_rois + 1)
                                ).difference(set(df.roi_num.unique())))
        total_missing = len(missing_rois)
        if total_missing != 0:
            # Too many missing values
            if total_missing > (df.shape[0] * .20):  # no more than 20%
                raise ValueError("Too much missing values.")
            for val in missing_rois:
                row = {'roi_num': val}
                columns = df.columns.to_list()
                columns.remove('roi_num')
                for colname in columns:
                    row[colname] = df[colname].mean()
                df = df.append(row, ignore_index=True)
                df = df.sort_values(by='roi_num').reset_index(drop=True)
            print(f"Interpolated {total_missing} missing value(s).")

    return df


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


# ============================== CLASSES ==============================
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
        self.filepath = self.output_path / \
            f"{self.subject_name}__{stats_filename}"
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
        from fsl.wrappers.fslmaths import fslmaths

        input_path = str(self.input_path)
        subject_stats = {
            'roi_num': [],
            'fa_mean': [], 'fa_std': [],
            'ad_mean': [], 'ad_std': [],
            'rd_mean': [], 'rd_std': [],
            'md_mean': [], 'md_std': [],
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

        # Mean Diffusivity (MD)
        # To compute the map of trace (TR) (equal to 3 times the mean diffusivity) 
        ret_code = subprocess.run([
            'TVtool', '-in', roi_path, '-tr', '-out', 'tmp_roi.nii.gz'
        ]).returncode
        if ret_code == 0:
            md_img = nib.load('tmp_roi.nii.gz').get_fdata() / 3
            roi_stats['md_mean'] = md_img.mean()
            roi_stats['md_std'] = md_img.std()
        return roi_stats


class DataLoader:
    from scipy import stats

    def __init__(self,
                 pre: List[Union[str, Path]],
                 post: List[Union[str, Path]],
                 n_rois: int,
                 fill_missing_roi: bool = True) -> None:
        """Data loader class for analysis using different tests

        Args:
            pre: list of paths of pre or group 1 csv data files.
            post: list of paths of post or group 2 csv data files.
            n_rois: Total number of ROIs present in the data.
                Make sure the ROI's numbering starts from 1 and consecutive.
            fill_missing_roi: if True, fill missing values with mean of the
                column.
        """
        # load data and set `roi_num` as index
        self.pre_dfs = self._load_data_safely(pre, fill_missing_roi, n_rois)
        self.post_dfs = self._load_data_safely(post, fill_missing_roi, n_rois)

        # set `roi_num` as index
        self.pre_dfs = {
            name: df.set_index('roi_num')
            for name, df in self.pre_dfs.items()
        }
        self.post_dfs = {
            name: df.set_index('roi_num')
            for name, df in self.post_dfs.items()
        }

        # Check if all dataframes are of same size
        size = [df.shape[0] == n_rois for name, df in self.pre_dfs.items()]
        _errmsg = f"All files must be of same size. Found {size}"
        assert all(size) == True, _errmsg

        self.n_rois = n_rois

    def _load_data_safely(self,
                          paths_list:  List[Union[str, Path]],
                          fill_missing_roi: bool,
                          n_rois: int):
        """Check and fix data before testing"""
        dfs = {}
        for p in paths_list:
            df = pd.read_csv(p)
            if fill_missing_roi:
                df = fill_missing_roi_values(df, n_rois)
            if df.shape[0] != n_rois:
                _errmsg = f"# ROIs in {p} != n_rois ({df.shape[0]} != {n_rois})."
                _errmsg += " set `fill_missing_roi = True` to fill missing values"
                raise ValueError(_errmsg)
            dfs[Path(p).stem] = df
        return dfs

    def paired_t_test(self,
                      var_name: str,
                      roi_num: Union[None, str, int, list],
                      roi_groups: Union[None, Dict[int, list]],
                      alpha: float = 0.05) -> dict:
        """Perform paired T-test on the given variable

        Args:
            var_name: column name of the variable to perform T-test.
            roi_num: ROI number i.e. specific brain region.
                [default is -1, means perform t-test on all regions one-by-one]
            roi_groups: Apply paried T-test on the groups of ROIs.
                There will be one p-value per group.
            alpha: Significance value
        """
        if (roi_num is not None) and (roi_groups is not None):
            raise AttributeError("Either pass `roi_num` or `roigrps`")

        elif roi_num is not None:
            if isinstance(roi_num, str) and (roi_num == 'all'):
                ret_dict = {}
                for i in range(1, self.n_rois + 1):
                    t_stat, p_value = self._one_roi_paired_t_test(
                        var_name, roi_num=i
                    )
                    ret_dict[i] = {
                        't_stat': t_stat,
                        'two_sided_p_value': p_value,
                        'p_value': p_value / 2,
                    }
                return ret_dict
            elif isinstance(roi_num, list):
                ret_dict = {}
                for i in roi_num:
                    t_stat, p_value = self._one_roi_paired_t_test(
                        var_name, roi_num=i
                    )
                    ret_dict[i] = {
                        't_stat': t_stat,
                        'two_sided_p_value': p_value,
                        'p_value': p_value / 2,
                    }
                return ret_dict
            else:
                try:
                    roi_num = int(roi_num)
                    if roi_num == 0:
                        raise ValueError("ROI number must be > 0.")
                except ValueError:
                    _errmsg = "`roi_num` must be an integer > 0 or `all`. "
                    _errmsg += f"Found roi_num = {roi_num}."
                    raise ValueError(_errmsg)
                t_stat, p_value = self._one_roi_paired_t_test(
                    var_name, roi_num=roi_num)
                return {roi_num: {
                        't_stat': t_stat,
                        'two_sided_p_value': p_value,
                        'p_value': p_value / 2,
                        }}

        elif roi_groups is not None:
            ret_dict = {}
            for group_num, group in roi_groups.items():
                ret = self._group_roi_paired_t_test(
                    var_name, group
                )
                group_t_stat, group_two_sided_p_value = ret
                group_p_value = group_two_sided_p_value / 2
                p_value = group_p_value / len(group)
                ret_dict[group_num] = {
                    'group_t_stat': group_t_stat,
                    'group_two_sided_p_value': group_two_sided_p_value,
                    'group_p_value': group_p_value,
                    'p_value': p_value
                }
            return ret_dict

    def _group_roi_paired_t_test(self, var_name: str, roi_group: list
                                 ) -> Tuple[float, float]:
        """Perform T-test on group of ROIs
        Args:
            var_name: Name of the variable (column of dataframe) to use.
            roi_group: list of roi numbers as a group.

        Returns:
            t-statistic, p-value
        """
        pre_values = [
            df.loc[roi_num, var_name]
            for df in self.pre_dfs.values() for roi_num in roi_group
        ]
        post_values = [
            df.loc[roi_num, var_name]
            for df in self.post_dfs.values() for roi_num in roi_group
        ]
        t_stat, two_sided_p_value = self.stats.ttest_rel(
            a=pre_values,
            b=post_values
        )
        return t_stat, two_sided_p_value

    def _one_roi_paired_t_test(self, var_name, roi_num
                               ) -> Tuple[float, float]:
        """Perform T-test on single ROI.
        Args:
            var_name: Name of the variable (column of dataframe) to use.
            roi_group: list of roi numbers as a group.

        Returns:
            t-statistic, p-value
        """
        pre_values = [
            df.loc[roi_num, var_name] for df in self.pre_dfs.values()
        ]
        post_values = [
            df.loc[roi_num, var_name] for df in self.post_dfs.values()
        ]
        t_stat, two_sided_p_value = self.stats.ttest_rel(
            a=pre_values,
            b=post_values
        )
        return t_stat, two_sided_p_value


class PairedTTest:
    from scipy import stats

    def __init__(self,
                 pre: List[Union[str, Path]],
                 post: List[Union[str, Path]],
                 n_rois: int,
                 fill_missing_roi: bool = True) -> None:
        """Perform Paired T-test
        Args:
            pre: list of paths of pre or group 1 csv data files.
            post: list of paths of post or group 2 csv data files.
            n_rois: Total number of ROIs present in the data.
                Make sure the ROI's numbering starts from 1 and consecutive.
            fill_missing_roi: if True, fill missing values with mean of the
                column.
        """
        # load data and set `roi_num` as index
        self.pre_dfs = self._load_data_safely(pre, fill_missing_roi, n_rois)
        self.post_dfs = self._load_data_safely(post, fill_missing_roi, n_rois)

        # set `roi_num` as index
        self.pre_dfs = {
            name: df.set_index('roi_num')
            for name, df in self.pre_dfs.items()
        }
        self.post_dfs = {
            name: df.set_index('roi_num')
            for name, df in self.post_dfs.items()
        }

        # Check if all dataframes are of same size
        size = [df.shape[0] == n_rois for name, df in self.pre_dfs.items()]
        _errmsg = f"All files must be of same size. Found {size}"
        assert all(size) == True, _errmsg

        # stack all subjects in an array

        self.n_rois = n_rois

    def run_roi(self,
                var_names: list,
                roi_num: Union[None, str, int, list],
                combine: bool = False
                ) -> dict:
        """Perform paired T-test on the given variable(s) collectively

        Args:
            var_names: list of column name(s) of the variable to
                perform T-test.
            roi_num: ROI number i.e. specific brain region.
                Pass an integer, 'all', or list of rois.
                Where `all` means perform t-test on all regions one-by-one.
            combine: If True, perform t-test on all variables and roi 
                combined. If False, perform t-test all variables and for 
                each roi separately
        Returns:
            dict containing t-statistic and p-value
        """
        # Pre and post data shapes must be (n_subjects, n_rois, n_variables)
        if isinstance(roi_num, str):
            if roi_num == 'all':
                pre_data, post_data = self._get_selected_roi_num_data(
                    roi_num,
                    var_names
                )
            else:
                try:
                    roi_num = int(roi_num)
                    if roi_num == 0:
                        raise ValueError("ROI number must be > 0.")
                    pre_data, post_data = self._get_selected_roi_num_data(
                        roi_num,
                        var_names
                    )
                except ValueError:
                    _errmsg = "`roi_num` must be an integer > 0 or `all`. "
                    _errmsg += f"Found roi_num = {roi_num}."
                    raise ValueError(_errmsg)
        elif isinstance(roi_num, (list, int)):
            if isinstance(roi_num, int) and (roi_num == 0):
                raise ValueError("ROI number must be > 0.")
            pre_data, post_data = self._get_selected_roi_num_data(
                roi_num,
                var_names)

        # number of subjects
        assert len(self.pre_dfs) == pre_data.shape[0]
        assert len(self.post_dfs) == post_data.shape[0]
        assert pre_data.shape == post_data.shape

        # Perform paired T-test
        if combine:
            t_stat, two_sided_p_value = self.stats.ttest_rel(
                a=pre_data.flatten(),
                b=post_data.flatten()
            )
            ret_dict = {
                't_stat': t_stat,
                'two_sided_p_value': two_sided_p_value,
                'p_value': two_sided_p_value / 2,
            }
        else:
            if isinstance(roi_num, list):
                ret_dict = {}
                for i, _roi in enumerate(roi_num):
                    t_stat, two_sided_p_value = self.stats.ttest_rel(
                        a=pre_data[:, i, :].flatten(),  # per ROI
                        b=post_data[:, i, :].flatten()
                    )
                    ret_dict[_roi] = {
                        't_stat': t_stat,
                        'two_sided_p_value': two_sided_p_value,
                        'p_value': two_sided_p_value / 2,
                    }
            elif roi_num == 'all':
                ret_dict = {}
                for i, _roi in enumerate(range(1, self.n_rois+1)):
                    t_stat, two_sided_p_value = self.stats.ttest_rel(
                        a=pre_data[:, i, :].flatten(),  # per ROI
                        b=post_data[:, i, :].flatten()
                    )
                    ret_dict[_roi] = {
                        't_stat': t_stat,
                        'two_sided_p_value': two_sided_p_value,
                        'p_value': two_sided_p_value / 2,
                    }
            else:
                # assert len(pre_data.shape) == 2
                t_stat, two_sided_p_value = self.stats.ttest_rel(
                    a=pre_data.flatten(),  # per ROI
                    b=post_data.flatten()
                )
                ret_dict = {
                    't_stat': t_stat,
                    'two_sided_p_value': two_sided_p_value,
                    'p_value': two_sided_p_value / 2,
                }
        return ret_dict

    def run_roi_groups(self,
                       var_names: list,
                       roi_groups: Union[None, Dict[int, list]]) -> dict:
        """Perform paired T-test on the given variable

        Args:
            var_names: list of column name(s) of the variable to
                perform T-test.
            roi_groups: Apply paried T-test on the groups of ROIs.
                There will be one p-value per group.
        """
        ret_dict = {}
        for grp_num, roi_list in roi_groups.items():
            results = self.run_roi(
                var_names=var_names,
                roi_num=roi_list,
                combine=True
            )
            results['group_p_value'] = results['p_value']
            # Normalize p_value
            results['p_value'] = results['group_p_value'] / len(roi_list)
            ret_dict[grp_num] = results
        return ret_dict

    def _load_data_safely(self,
                          paths_list:  List[Union[str, Path]],
                          fill_missing_roi: bool,
                          n_rois: int):
        """Check and fix data before testing"""
        dfs = {}
        for p in paths_list:
            df = pd.read_csv(p)
            try:
                if fill_missing_roi:
                    df = fill_missing_roi_values(df, n_rois)
            except ValueError as e:
                print(e)
                raise ValueError(f"Error @ `{p}`")
                
            if df.shape[0] != n_rois:
                _errmsg = f"# ROIs in {p} != n_rois ({df.shape[0]} != {n_rois})."
                _errmsg += " set `fill_missing_roi = True` to fill missing values"
                raise ValueError(_errmsg)
            dfs[Path(p).stem] = df
        return dfs

    def _get_selected_roi_num_data(
        self,
        roi_num: Union[str, int, list],
        var_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get selected ROIs data from subject dataframes in np.array format
        """
        if roi_num == 'all':
            pre_data = np.array([
                s[var_names].values
                for s in self.pre_dfs.values()
            ])
            post_data = np.array([
                s[var_names].values
                for s in self.post_dfs.values()
            ])
        else:
            pre_data = np.array([
                s[var_names].loc[roi_num].values
                for s in self.pre_dfs.values()
            ])
            post_data = np.array([
                s[var_names].loc[roi_num].values
                for s in self.post_dfs.values()
            ])
        return pre_data, post_data
