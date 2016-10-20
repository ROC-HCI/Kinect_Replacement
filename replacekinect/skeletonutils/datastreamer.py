import h5py
import os
import numpy as np

# Data stream generator to flow the frames from the h5 file
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

# Data stream generator to flow the frames from the h5 file
# Shuffle the dataset and reset once the generator ends
def data_stream_shuffle_reset(datafile,datrng,batchsize=128):
    if not os.path.isfile(datafile):
        raise IOError('File not found')
    assert type(datrng) in {list,tuple} and len(datrng) == 2 and datrng[0]<=datrng[1]
    while True:
        with h5py.File(datafile,'r') as f:
            allsubj = f.keys()
            np.random.shuffle(allsubj)
            for subj in allsubj:
                if int(subj)>=datrng[0] and int(subj)<datrng[1]:
                    allvids = f[subj].keys() 
                    np.random.shuffle(allvids)
                    for vid in allvids:
                        v = subj+'/'+vid+'/video_frames'
                        s = subj+'/'+vid+'/joints'
                        ind = 0
                        N = np.size(f[v],axis=0)
                        while ind<N:
                            end = min(ind+batchsize,N)
                            i = np.arange(end-ind)
                            np.random.shuffle(i)
                            frames,joints = f[v][ind:end,:,:,:],f[s][ind:end,:]
                            yield frames[i,:,:,:],joints[i,:]
                            ind+=batchsize
