import os
import h5py
import numpy as np

from skeletonutils import skelviz_mayavi as sviz
from skeletonutils import data_stream_shuffle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Module for testing neural network to replace kinect')
parser.add_argument('datafile',help='Full path of the data (h5) file')
parser.add_argument('-m',dest='modelid',type=int,default=1,\
    help='ID of the preset model to be loaded')
parser.add_argument('--weightfile',dest='weightfile',default='weightfile.h5',\
    help='Weight filename')
args = parser.parse_args()

#datafile = '/scratch/mtanveer/automanner_dataset.h5'
datafile = '/Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5'
trainset = (34,55)
testset = (55,63)

if args.modelid==1:
    from learningtools.preset_models import original
    cnnmodel,model=original(True,args.weightfile,False)

# Create batch over test dataset and compute loss
print 'loss calculation'
for frames,joints in data_stream_shuffle(args.datafile,testset):
    # Get the prediction
    newinput = cnnmodel.predict(frames[:1,:,:,:])
    newoutput = np.insert(model.predict(newinput)[0,:],0,[0,0])    
    plt.imshow(np.transpose(frames[0,:,:,:],axes=[1,2,0]))
    plt.ion()
    plt.show()
    sviz.drawskel(newoutput)


