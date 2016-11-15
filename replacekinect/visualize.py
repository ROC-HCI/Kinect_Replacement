import os
import h5py
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from train import parse_modelid
from skeletonutils import data_stream_shuffle

plotnames = ['original','cnnblk4','cnnblk3','tunable','dd','dd_bn','dd_bn_rg',\
'cnnblk3_dd_bn_rg','res_bn_rg']

def vizloss(folderpath):
    files = os.listdir(folderpath)
    n = len(files)
    colors = iter(plt.cm.Set1(np.linspace(0,1,n)))
    for afile in files:
        if afile.startswith('kr_pre'):
            train_loss=[]
            test_loss=[]
            filename = os.path.join(folderpath,afile)
            with open(filename,'r') as f:
                print 'reading ...',filename
                for i in f:
                    if i.startswith('# of Data fed'):
                        parts = i.split('Loss:')
                        train_loss.append(math.log(float(parts[1][:-6])))
                        test_loss.append(math.log(float(parts[2])))
            plt.figure(1)
            c = next(colors)
            plotlabel = plotnames[int(afile[6:-8])-1]+' ('+afile[6:-8]+')'
            plt.plot(train_loss,label=plotlabel,c=c,linewidth=1.5)
            plt.figure(2)
            plt.plot(test_loss,label=plotlabel,c=c,linewidth=1.5)
    plt.figure(1)
    plt.title('Training Loss')
    plt.legend()
    plt.figure(2)
    plt.title('Test Loss')
    plt.legend()
    plt.show()

# Visualize some sample prediction data
def vizsample(data_gen,cnnmodel,model):
    # Locally importing in order to reduce loading time for others
    from skeletonutils import skelviz_mayavi as sviz
    # Create batch over test dataset and compute loss
    for frames,joints in data_gen:
        # Get the prediction
        newinput = cnnmodel.predict(frames[:1,:,:,:])
        outjoints = model.predict(newinput)[0,:]
        # Append time and frame number for skeleton plotter
        newoutput = np.insert(outjoints,0,[0,0])
        plt.imshow(np.transpose(frames[0,:,:,:],axes=[1,2,0]))
        # Get the prediction
        newinput = cnnmodel.predict(frames[:1,:,:,:])
        newoutput = np.insert(model.predict(newinput)[0,:],0,[0,0])
        plt.imshow(np.transpose(frames[0,:,:,:],axes=[1,2,0]))
        plt.ion()
        plt.show()
        sviz.drawskel(newoutput)
        inp = raw_input('Press Enter to continue (or "quit" to exit) ...') 
        if inp == 'quit':
            break

# Calculates all the necessary to plot an Accuracy vs. MSE plot
def save_acc_mse(data_gen,out_prefix,cnnmodel,model):
    # Locally importing in order to reduce loading time for others
    mse=[]
    batch_count = 0
    for frames,joints in data_gen:
        # Get the prediction
        newinput = cnnmodel.predict(frames)
        outjoints = model.predict(newinput)
        mse.extend(np.mean((joints-outjoints)**2.,axis=1).tolist())
        print 'Batch:',batch_count
        batch_count+=1
    # Save the histogram for total mse
    acc,mse = np.histogram(mse,1000)
    acc = np.cumsum(acc)
    acc = acc/float(np.max(acc))
    if len(acc)!=len(mse):
        mse = mse[1:]
    with h5py.File(out_prefix+'hist.h5','w') as f:
        f['/accuracy'] = acc
        f['/mse_bins'] = mse

# Draw Accuracy vs. MSE plot
def show_acc_mse(folderpath,logmse=False):
    allfiles = [item for item in os.listdir(folderpath) if item.endswith('h5')]
    print 'Looking into folder:',folderpath
    if not allfiles:
        print 'folder empty:',folderpath
        return
    # Plot the accuracy vs mse graph
    colors = iter(plt.cm.Set1(np.linspace(0,1,len(allfiles))))
    plt.figure()
    for afile in allfiles:
        file_ = os.path.join(folderpath,afile)
        with h5py.File(file_) as f:
            c = next(colors)
            modelid = int(afile[3:4])-1
            if logmse:
                mse = np.log(f['/mse_bins'][:])
            else:
                mse = f['/mse_bins'][:]
            plt.plot(mse,f['/accuracy'][:],c=c,linewidth=1.5,\
                label=plotnames[modelid]+'('+str(modelid+1)+')')
    if logmse:
        plt.xlabel('Log of Mean Squared Error')
    else:
        plt.xlabel('Log of Mean Squared Error')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    # Argument parser
    parser = argparse.ArgumentParser('Module for testing neural network to replace kinect')
    parser.add_argument('command',choices=['loss','acc_mse_save','acc_mse_show','sample'],\
        help='Commands specifying what to do (%(choices)s)')
    parser.add_argument('--data',dest='datafile',help='Full path of the data (h5) file')
    parser.add_argument('--weight',dest='weightfile',\
        help='Trained weight filename for the preset model. The filename will determine \
        which preset model will be loaded.')
    parser.add_argument('--losspath',dest='losspath',\
        help='Path to the folder where loss files ("kr_pre*") are located')
    parser.add_argument('--histpath',dest='histpath',default='../Results/result_hist/',\
        help='Path to the folder where the *hist.h5 are located')
    parser.add_argument('--vggweightfile',dest='vggweightfile',default='vgg16_weights.h5',\
        help='Weight filename (default: %(default)s)')
    parser.add_argument('--prefout',dest='prefout',default='',\
        help='Prefix of output files for acc_mse_save command')
    parser.add_argument('--log_mse',dest='logmse',action='store_true',default=False,\
        help='Use the logarithm of mean-squared-error as the x axis of acc-mse plot')
    args = parser.parse_args()

    # Perform action based on the command
    if args.command == 'sample' or args.command == 'acc_mse_save':
        if not args.datafile or not args.weightfile or not args.vggweightfile:
            print 'The following arguments are necessary to sample or plot error:'
            print 'Full path of the data file (--data)'
            print 'Trained weight file (--weight)'
            print 'Vgg weight file (--vggweightfile)', '(optional)'
            return
        # datafile = '/scratch/mtanveer/automanner_dataset.h5'
        # datafile = '/Users/itanveer/Data/ROCSpeak_BL/allData_h5/automanner_dataset.h5'
        with h5py.File(args.datafile) as f:
            trainset = f.attrs['training_range'].tolist()
            testset = f.attrs['test_range'].tolist()

        # Select correct preset model from the weightfile naming convention
        path,file = os.path.split(args.weightfile)
        if not file.startswith('pre') or not file.endswith('weightfile.h5') or \
            not int(file[3:4]) in range(1,10) or not '_' in file:
            print 'Weightfile name is not recognized (gotta start with pre and end with \
            weightfile.h5)'
            return
        m = file.index('_')
        modelid = int(file[3:m])
        cnnmodel,model=parse_modelid(modelid,True,args.weightfile,False,args.vggweightfile)
        # Choose the correct action
        if args.command=='sample':
            data_gen = data_stream_shuffle(args.datafile,testset)
            vizsample(data_gen,cnnmodel,model)
        elif args.command=='acc_mse_save':
            data_gen = data_stream_shuffle(args.datafile,testset,batchsize=1024)
            save_acc_mse(data_gen,args.prefout,cnnmodel,model)
    elif args.command == 'loss':
        if not args.losspath:
            print 'The following argument is necesary for plotting loss function:'
            print 'Path to the folder where loss files ("kr_pre*") are located (--losspath)'
            return
        vizloss(args.losspath)
    elif args.command == 'acc_mse_show':
        if not args.histpath:
            print 'The following argument is necesary for plotting loss function:'
            print 'Path to the folder where the *hist.h5 are located (--histpath)'
            return
        show_acc_mse(args.histpath,args.logmse)
    else:
        print 'Command not recognized'
        

if __name__=='__main__':
    main()


