# Train the original preset model after loading the 
# weights saved in previous iterations
python train.py '/scratch/mtanveer/automanner_dataset.h5' --load_weights
# Train the original preset model WITHOUT loading the 
# weights saved in previous iterations
python train.py '/scratch/mtanveer/automanner_dataset.h5'
# Train the preset model with only 4 CNN blocks
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 2
# Train the preset model with only 3 CNN blocks
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 3

# Data file in my local folder structure
# python train.py /Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5 --load_weights
