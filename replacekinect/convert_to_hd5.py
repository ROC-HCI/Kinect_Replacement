import h5py
import cPickle as cp
import numpy as np
import os
from skeletonutils import skelio

datapath = '/Users/itanveer/Data/ROCSpeak_BL/pickled_videos_skeletons/'
f = h5py.File('automanner_dataset.h5','w')

for afile in os.listdir(datapath):
    if not afile.endswith('pkl'):
        continue
    filename = os.path.join(datapath,afile)
    data = cp.load(open(filename,'r'))
    framedata = np.asarray(data['frame'],dtype=np.uint8)
    skeldata = skelio.shiftorigin(np.asarray(data['joints'],dtype=np.float32))[0]
    vtime = np.expand_dims(np.array(data['video_time'],dtype=np.int),axis=1)

    # Converting from BGR to RGB
    # Converting from nxMxNxc to nxcxMxN
    framedata = np.transpose(framedata[:,:,:,::-1],axes=[0,3,1,2])
    f.create_dataset(afile[:2]+'/'+afile[3]+'/'+'video_frames',
        data = framedata)
    f.create_dataset(afile[:2]+'/'+afile[3]+'/'+'joints',
        data = skeldata[:,2:])
    f.create_dataset(afile[:2]+'/'+afile[3]+'/'+'time_stamps',
        data = np.hstack((skeldata[:,:2],vtime)))
    print afile,'... Done'
f.close()
