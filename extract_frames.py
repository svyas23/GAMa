import cv2
import numpy as np
import os
import sys

import params as params


def extract_frames(vid, dst):
    print(vid)
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    print(image.shape)
    print(success)
    count = 0
    while success:
        image = cv2.resize(image, (144, 256)) # original size 720x1280
        img_path = os.path.join(dst, '%04d.png' % count)
        print(img_path)
        cv2.imwrite(img_path, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    return count

def master(v_list, mode):

    #pack_path =  params.raw_videos_train  
    #pack_path =  params.raw_videos_val
    
    videos = open(v_list, 'r')

    fname = mode + '_4.list'
    fp = open(fname, 'w')
    
    pack_path =  params.raw_videos_train
    base_path = '/home/c3-0/shruti/data/BDD_frames/frames144_256/train/'
        
    if mode == 'val':
        pack_path =  params.raw_videos_val
        base_path = '/home/c3-0/shruti/data/BDD_frames/frames144_256/val/'

    cnt = 0
    for sample in videos:
        sample = sample.rstrip('\n')
        
        v_name, num_frames = sample.split()
          
        if int(num_frames) < 1200:
            print(num_frames)
            print(v_name)
            cnt += 1
        
            sample_path = os.path.join(pack_path, v_name+'.mov')
    
            dst = os.path.join(base_path, v_name+'.mov')
    
            num_frames = extract_frames(sample_path, dst)
            fp.write('{} {}\n'.format(v_name, num_frames))
        else:
            fp.write('{}\n'.format(sample))
            pass
    
    print(cnt)

    fp.close()
    videos.close()

if __name__ == '__main__':
    v_list = None
    if len(sys.argv) > 1:
        mode = str(sys.argv[1])
    else:
        print('mode missing!')
        exit(0)

    v_list = mode + '_3.list'

    master(v_list, mode)

