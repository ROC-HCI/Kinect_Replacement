import h5py
import sys
import numpy as np
import argparse
import itertools as it
from skeletonutils import data_stream_shuffle

# Argument parser
parser = argparse.ArgumentParser('Module for training neural network to replace kinect')
parser.add_argument('datafile',help='Full path of the data (h5) file')
parser.add_argument('-m',dest='modelid',type=int,default=1,\
    help='ID of the preset model to be loaded')
parser.add_argument('-i',dest='nb_iter',type=int,default=10,\
    help='Total number of iterations')
parser.add_argument('-b',dest='batch_size',type=int,default=128,help='Batch Size')
parser.add_argument('--load_weights',dest='load_weights',action='store_true',\
    default=False,help='Load previously saved weights')
parser.add_argument('--stop_summary',dest='stop_summary',action='store_true',\
    default=False,help='Stops printing the model summary before training')
parser.add_argument('--weightfile',dest='weightfile',default='weightfile.h5',\
    help='Weight filename')
parser.add_argument('--out_prefix',dest='out_prefix',default='',\
    help='A prefix for the output weight file')
args = parser.parse_args()

# Training test split
trainset = (34,55)
testset = (55,63)

if args.modelid==1:
    from learningtools.preset_models import original
    cnnmodel,model=original(args.load_weights,args.weightfile,args.stop_summary)

# Create batch and feed the fully connected neural network
count = 0
# Test data generator (Never ending)
test_stream = it.cycle(data_stream_shuffle(args.datafile,testset,batchsize=args.batch_size))
print 'Starting Training ... '
for iter in range(args.nb_iter):
    # Flow data from the training data stream
    for frames, joints in data_stream_shuffle(args.datafile,trainset,batchsize=args.batch_size):
        newinput = cnnmodel.predict(frames) # pass through CNN layers
        tr_loss = model.train_on_batch(newinput,joints) # train on batch
        # get next test data
        tst_frame,tst_joints = next(test_stream)
        # Pass through the model pipeline
        tst_frame = cnnmodel.predict(tst_frame)
        tst_loss = model.test_on_batch(tst_frame,tst_joints)
        # print status
        count+=len(frames)
        print '# of Data fed:',count, 'Mean Train Loss:',np.mean(tr_loss),\
            'Test Loss:',tst_loss[0]
        sys.stdout.flush()
    print 'iteration:',iter, 'saving weights ...',
    count = 0
    # Save the model
    model.save_weights(args.out_prefix+args.weightfile)
    print 'done.'
print 'Training Finished'


