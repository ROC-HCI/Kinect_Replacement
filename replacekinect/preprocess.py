from __future__ import print_function
import os
import cv2
import numpy as np
import cPickle as cp
from skeletonutils import skelviz_mayavi,skelio


# Data location
datapath = '/Users/itanveer/Data/ROCSpeak_BL'
frame_stride=10 # Frames to skip = frame_stride-1


# for afile
skelpath = os.path.join(datapath,'allSkeletons')
for afile in os.listdir(skelpath):
    dirpath,filename = os.path.split(afile)
    if not afile.endswith('csv'):
        continue

    # Open video and skeleton file
    vidfile = os.path.join(datapath,'allvideos',afile[:-3]+'mp4')
    skelfile = os.path.join(datapath,'allSkeletons',afile)
    data,header = skelio.readdatafile(skelfile)
    cap = cv2.VideoCapture(vidfile)

    # Reset counters and accumulators
    frame_accum = []
    skel_accum = []
    vtime_accum = []
    frame_nb=0  # Start frame

    # Loop through
    max_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    while frame_nb<max_frames and frame_nb<data[-1,0]:
        # marking a frame from retrieval
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_nb)
        cap.grab()
        vidtime = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        framepos = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        
        # retreiving skeleton frame
        skeleton_index = np.argmin(np.abs(data[:,1]-vidtime))
        skel_frame = data[skeleton_index,:]

        # retrieving video frame
        ret,frame = cap.retrieve()
        frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

        # Store frames and skeleton
        if ret:
            frame_accum.append(frame)
            skel_accum.append(skel_frame)
            vtime_accum.append(vidtime)
        else:
            frame_nb+=1
            continue

        # print status
        print(afile,'idx:',frame_nb,'vframe:',\
            '{0:.2f}'.format(framepos),'vtime:',int(vidtime),\
            'sindx:',skeleton_index,sep='\t')

        # Increment frame count
        frame_nb+=frame_stride

    # Save data for current file
    cp.dump({'frame':frame_accum,'joints':skel_accum,\
        'video_time':vtime_accum},open(filename[:-3]+'pkl','wb'))

        # # Draw the retrieved frames
        # cv2.imshow('test',frame)
        # cv2.waitKey(10)
        # skelviz_mayavi.drawskel(skel_frame)
        # if cv2.waitKey(1)&0xff == ord('q'):
        #     break
        