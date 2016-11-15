import h5py
import os
import numpy as np
import itertools as it
from keras.applications.imagenet_utils import preprocess_input

# Data stream generator with better randomization
# If preprocess is True, the frame is mean-centered and converted to BGR
# This is necessary for working with VGG model
def data_stream_shuffle(datafile,datrng,batchsize=128,preprocess=True):
    # Input check and flatten until the lowest layer
    if not os.path.isfile(datafile):
        raise IOError('File not found')
    assert type(datrng) in {list,tuple} and len(datrng) == 2 and datrng[0]<=datrng[1]
    with h5py.File(datafile,'r') as f:
        # Flatten by adding all nodes (until the lowest layer) in a list
        all_recent_parents = []
        for subj in f:
            if int(subj)>=datrng[0] and int(subj)<datrng[1]:
                for vid in f[subj]:
                    all_recent_parents.append(subj+'/'+vid)

        # rem is the remainder child(ren) index of the current parent
        rem = {aparent:range(len(f[aparent+'/video_frames'][:])) \
            for aparent in all_recent_parents}
        batch_v = []
        batch_j = []
        while rem:
            # sample a parent
            aparent = np.random.choice(rem.keys())
            if rem[aparent]:
                # sample and update the remainder
                x = np.random.choice(rem[aparent])
                rem[aparent].remove(x)
                batch_v.append(f[aparent+'/video_frames'][x,:,:,:])
                batch_j.append(f[aparent+'/joints'][x,:])
            else:
                del rem[aparent]
            if len(batch_v)==batchsize or (not rem and len(batch_v)>0):
                # Preprocess the frames
                if preprocess:
                    frames = preprocess_input(np.array(batch_v,dtype='float64'))
                yield frames, np.array(batch_j,dtype='float64')
                batch_v,batch_j = [],[]
