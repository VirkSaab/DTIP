# Process one subject
#dtip process /media/virk/MyEHDD/PGI_ITS_RAW/RAW/1_1 -o /media/virk/MyEHDD/ITS

#------------- FROM RAW DATA TO ANALYSIS COMPLETE PIPELINE COMMANDS -----------
# Process multiple subjects (some subjects)
dtip process-multi /media/virk/MyEHDD/PGI_ITS_RAW/RAW -o /media/virk/MyEHDD/ITS/

# FSL to DTI-TK conversion
dtip fsl-dtitk-multi /media/virk/MyEHDD/ITS/3_process/ -o /media/virk/MyEHDD/ITS/4_register

# Bootstrap initial template (using 35 subjects. Delete unpaired subjects after this)
dtip bootstrap-template /media/virk/MyEHDD/ITS/4_register /media/virk/MyEHDD/templates/JHU_fa_128.nii.gz

# Multi subjects registration (only register if diffeomorphic is required. Otherwise, affine is already created by bootstrapped_template command)
dtip register-multi -o /media/virk/MyEHDD/ITS/4_register/ /media/virk/MyEHDD/ITS/4_register/ -btp /media/virk/MyEHDD/ITS/bootstrapped_template.nii.gz

# Template to subject space for multiple subjects
dtip template-to-subject-multi /media/virk/MyEHDD/ITS/4_register/ /media/virk/MyEHDD/templates/JHU_pcl_128.nii.gz -tt diffeo

# Compute ROI for a subject
dtip compute-stats /media/virk/MyEHDD/ITS/4_register/1_1/dti_dtitk.nii.gz -tn JHU_pcl_128_dti_space.nii.gz -o /media/virk/MyEHDD/ITS/5_analysis
### For multiple subjects
dtip compute-stats-multi /media/ubuntu/MyEHDD/ITS22/4_register/ -sn dti_dtitk.nii.gz -tn JHU_pcl_128_shifted_diffeo_dti_space.nii.gz -o /media/ubuntu/MyEHDD/ITS22/5_analysis
#------------------------------------------------------------------------------