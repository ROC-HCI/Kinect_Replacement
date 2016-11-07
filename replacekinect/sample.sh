# Training Options
###################

# Train the original preset model after loading the 
# weights saved in previous iterations
python train.py '/scratch/mtanveer/automanner_dataset.h5' --load_weights
# Train the original preset model WITHOUT loading the 
# weights saved in previous iterations
python train.py '/scratch/mtanveer/automanner_dataset.h5'
# Train the second preset model that has only 4 CNN blocks
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 2
# Use 'preset2' as a prefix to the saved weight file
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 2 --out_prefix preset2_
# While saving the weight files, augment the number of iterations so that they don't overwrite one another
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 2 --out_prefix preset2_ --prefit 
# Specify the weight filename to load when in previous iteration it was named 'preset2_weightfile.h5'
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 2 --load_weights --weightfile preset2_weightfile.h5
# Train the third preset model that has 3 CNN blocks
python train.py '/scratch/mtanveer/automanner_dataset.h5' -m 3

# Data file in my local folder structure
# python train.py /Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5 --load_weights

# Test Options
##############
# Check loss functions
python visualize.py loss --losspath /Users/itanveer/Devel/Kinect_Replacement/Results/result_3d
# check sample predictions
python visualize.py sample --data /Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5 --weight /Users/itanveer/Devel/Kinect_Replacement/Results/result_3d/pre4_weightfile.h5
# Perform necessary calculations for MSE_ACC plot
python visualize.py mse_acc_save --data /scratch/mtanveer/automanner_dataset.h5 --weight pre2_weightfile.h5 --prefout 'pre2_'
# Show the MSE_ACC plot (if the outputs of mse_acc_save command is saved to ../Results/result_hist/)
python visualize.py mse_acc_show --histpath ../Results/result_hist/