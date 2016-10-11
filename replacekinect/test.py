import os
import h5py
import numpy as np
from keras.applications import vgg16
from keras.layers import Flatten, Dense, Input
from keras.models import Model

datafile = '/scratch/mtanveer/automanner_dataset.h5'
trainset = (34,55)
testset = (55,63)

# Data stream generator to flow the frames from the h5 file
# TODO: Make it randomized
def data_stream(datafile,datrng,batchsize=128):
    if not os.path.isfile(datafile):
        raise IOError('File not found')
    assert type(datrng) in {list,tuple} and len(datrng) == 2 and datrng[0]<=datrng[1]
    with h5py.File(datafile,'r') as f:
        for subj in f:
            if int(subj)>=datrng[0] and int(subj)<datrng[1]:
                for vid in f[subj]:
                    v = subj+'/'+vid+'/video_frames'
                    s = subj+'/'+vid+'/joints'
                    ind = 0
                    N = np.size(f[v],axis=0)
                    while ind<N:
                        end = min(ind+batchsize,N)
                        yield f[v][ind:end,:,:,:],f[s][ind:end,:]
                        ind+=batchsize

# Vgg model without fully connected layer
vggmodel = vgg16.VGG16(include_top=False)
vggmodel.summary()

# create fully connected layer
fc_input = Input(shape=(512,5,10))
x = Flatten(name='flatten')(fc_input)
x = Dense(1024, activation='relu',name='fc1')(x)
x = Dense(1024, activation='relu',name='fc2')(x)
x = Dense(60,activation='linear',name='predictions')(x)
fcmodel = Model(fc_input,x)
#fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',metrics=['accuracy'])
fcmodel.load_weights('weightfile.h5')
fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',metrics=['accuracy'])
print 'Model Loaded'

# Create batch over test dataset and compute loss
print 'loss calculation'
for frames,joints in data_stream(datafile,testset):
    newinput = vggmodel.predict(frames)
    loss = fcmodel.test_on_batch(newinput,joints)
    print loss
