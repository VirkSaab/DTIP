# Process one subject
#dtip process /media/virk/MyEHDD/PGI_ITS_RAW/RAW/1_1 -o /media/virk/MyEHDD/ITS

# Process multiple subjects (some subjects)
dtip process-multi /media/virk/MyEHDD/PGI_ITS_RAW/RAW -o /media/virk/MyEHDD/ITS/

# Bootstrap initial template
dtip bootstrap-template ITS/3_process templates/JHU_fa_128.nii.gz

# Multi subjects registration
dtip register-multi -o /media/virk/MyEHDD/ITS/4_register/ /media/virk/MyEHDD/ITS/3_process/ -btp /media/virk/MyEHDD/ITS/bootstrapped_template.nii.gz

# Template to subject space for multiple subjects
dtip template-to-subject-multi ITS/4_register/ templates/JHU_pcl_128.nii.gz

