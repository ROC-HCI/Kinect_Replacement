import h5py
import sys
import numpy as np
import argparse
import itertools as it
from skeletonutils import data_stream_shuffle

def main():
    # Argument parser
    parser = argparse.ArgumentParser('Module for training neural network to replace kinect')
    parser.add_argument('datafile',help='Full path of the data (h5) file')
    parser.add_argument('-m',dest='modelid',type=int,choices=range(1,10),default=1,\
        help='ID of the preset model to be loaded. choices=%(choices)s. \
        ID = 1 is the original preset model \
        with 5 blocks of CNN from VGG16, two FC layers with 1024 relu neurons, and one FC \
        layer with 60 linear neurons. ID = 2 is similar, except 4 blocks of CNN from vgg. \
        ID = 3 has 3 blocks of CNN. ID = 4 has a block of tunable CNN. In ID = 5, number \
        of dense layers and neurons are doubled. ID = 6 doubledense with batch normalization \
        ID = 7 doubledense, BN, and regularization (0.01 l1). ID = 8 has 3 cnn blks, dd, \
        BN, regularization. ID = 9 has residual doubledense with BN, rg')
    parser.add_argument('-i',dest='nb_iter',type=int,default=10,\
        help='Total number of iterations. (default: %(default)s)')
    parser.add_argument('-b',dest='batch_size',type=int,default=128,\
        help='Batch Size (default: %(default)s)')
    parser.add_argument('--load_weights',dest='load_weights',action='store_true',\
        default=False,help='Load previously saved weights (default: %(default)s)')
    parser.add_argument('--stop_summary',dest='stop_summary',action='store_true',default=False,\
        help='Stops printing the model summary before training (default: %(default)s)')
    parser.add_argument('--weightfile',dest='weightfile',default='weightfile.h5',\
        help='Weight filename (default: %(default)s)')
    parser.add_argument('--vggweightfile',dest='vggweightfile',default='vgg16_weights.h5',\
        help='Weight filename (default: %(default)s)')
    parser.add_argument('--out_prefix',dest='out_prefix',default='',\
        help='A prefix for the output weight file (default: <Empty String>')
    parser.add_argument('--prefit',dest='prefit',action='store_true',default=False,\
        help='Add iteration as a prefix to the output weight filename so that \
        the weight files do not overwrite themselves (default: %(default)s)')

    args = parser.parse_args()

    # Training test split
    trainset = (34,55)
    testset = (55,63)

    # Parse the modelid
    cnnmodel,model = parse_modelid(args.modelid,args.load_weights,\
        args.weightfile,args.stop_summary,args.vggweightfile)

    # Create batch and feed the fully connected neural network
    count = 0
    # Test data generator (Never ending)
    test_stream = it.cycle(data_stream_shuffle(args.datafile,testset,\
        batchsize=args.batch_size))
    print 'Starting Training ... '
    for iter_ in range(args.nb_iter):
        # Flow data from the training data stream
        for frames, joints in data_stream_shuffle(args.datafile,trainset,\
            batchsize=args.batch_size):
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
        print 'iteration:',iter_, 'saving weights ...',
        count = 0
        # Save the model
        if args.prefit:
            model.save_weights(args.out_prefix+str(iter_)+args.weightfile)
        else:
            model.save_weights(args.out_prefix+args.weightfile)
        print 'done.'
    print 'Training Finished'

def parse_modelid(modelid,load_weights,weightfile,stop_summary,\
    vggweightfile):
        # Parse Modelid
    if modelid==1:
        from learningtools.preset_models import original
        cnnmodel,model=original(load_weights,weightfile,stop_summary)
    elif modelid==2:
        from learningtools.preset_models import lesscnn
        cnnmodel,model=lesscnn(load_weights,weightfile,vggweightfile,\
            stop_summary,4)
    elif modelid==3:
        from learningtools.preset_models import lesscnn
        cnnmodel,model=lesscnn(load_weights,weightfile,vggweightfile,\
            stop_summary,3)
    elif modelid==4:
        from learningtools.preset_models import tunable
        cnnmodel,model=tunable(load_weights,weightfile,vggweightfile,\
            stop_summary)
    elif modelid==5:
        from learningtools.preset_models import doubledense
        cnnmodel,model=doubledense(load_weights,weightfile,\
            stop_summary)
    elif modelid==6:
        from learningtools.preset_models import doubledense_bn
        cnnmodel,model=doubledense_bn(load_weights,weightfile,\
            stop_summary)
    elif modelid == 7:
        from learningtools.preset_models import doubledense_bn_rg
        cnnmodel,model=doubledense_bn_rg(load_weights,weightfile,\
            stop_summary)
    elif modelid == 8:
        from learningtools.preset_models import lesscnn_dd_bn_rg
        cnnmodel,model=lesscnn_dd_bn_rg(load_weights,weightfile,\
            vggweightfile,stop_summary,3)
    elif modelid == 9:
        from learningtools.preset_models import residual_bn_rg
        cnnmodel,model=residual_bn_rg(load_weights,weightfile,\
            stop_summary)
    else:
        raise ValueError('Model ID not recognized')
    return cnnmodel,model

if __name__=='__main__':
    main()