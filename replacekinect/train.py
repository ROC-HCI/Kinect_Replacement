import h5py
import numpy as np
import itertools as it
from keras.applications import vgg16
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from skeletonutils import data_stream_shuffle

datafile = '/scratch/mtanveer/automanner_dataset.h5'
#datafile = '/Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5'
trainset = (34,55)
testset = (55,63)
nb_iter = 100
batch_size = 128
load_weights = True
weightfile = 'weightfile.h5'
out_prefix = ''

# Vgg model without fully connected layer
vggmodel = vgg16.VGG16(include_top=False)

# create fully connected layer
fc_input = Input(shape=(512,5,10))
x = Flatten(name='flatten')(fc_input)
x = Dense(1024, activation='relu',name='fc1')(x)
x = Dense(1024, activation='relu',name='fc2')(x)
x = Dense(60,activation='linear',name='predictions')(x)
fcmodel = Model(fc_input,x)

# Load the model weights if instructed
if load_weights:
    fcmodel.load_weights(weightfile)

fcmodel.compile(loss='mean_squared_error',optimizer='adagrad',metrics=['accuracy'])
fcmodel.summary()
print 'Fully-Connected model prepared'

# Create batch and feed the fully connected neural network
count = 0
test_stream = it.cycle(data_stream_shuffle(datafile,testset,batchsize=1))
print 'Starting Training ... '
for iter in range(nb_iter):
    for frames, joints in data_stream_shuffle(datafile,trainset,batchsize=batch_size):
        newinput = vggmodel.predict(frames) # pass through CNN layers
        tr_loss = fcmodel.train_on_batch(newinput,joints) # train on batch
        # test on train data
        tst_frame,tst_joints = next(test_stream) 
        tst_frame = vggmodel.predict(tst_frame)
        tst_loss = fcmodel.test_on_batch(tst_frame,tst_joints)
        count+=len(frames)
        print '# of Data fed:',count, 'Mean Train Loss:',np.mean(tr_loss),\
            'Test Loss:',tst_loss[0]
    print 'saving weights ...',
    fcmodel.save_weights(out_prefix+weightfile)
    print 'done.'
print 'Training Finished'


