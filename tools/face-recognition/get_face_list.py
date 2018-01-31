# coding:utf-8

import numpy as np
import sys,os

#dir_root = '/home/sdc1/zf/datasets/face-recognition/VGGFace2/'
img_root = sys.argv[1]
img_list = sys.argv[2]

#overlap_name = ['0166921','1056413','1193098']
overlap_name =[]
list = []
files = os.listdir(img_root)
fid = open(img_list,'w')
cnt = 0
for file in files:
    m = os.path.join(img_root,file)
    if(os.path.isdir(m)):
        print('collecting the %d th folder (total %d)' % (cnt+1,len(files)))
        h = os.path.split(m)
        if(h[1] in overlap_name):
            continue
        img_name = os.listdir(m)
        for im in img_name:
            fileName = os.path.join(m,im)
            if (os.path.isdir(fileName)):
                continue
            fid.write('%s %d\n' %(fileName,cnt))
        cnt = cnt+1
        list.append(h[1])
fid.close()
print('generate image list finished')
