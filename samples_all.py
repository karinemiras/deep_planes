import os
import glob
import shutil
import numpy as np

datasets_names = ['Butterfly',
                  'Giraffe',
                  'Plane']

datasets_categs = ['b', 'g', 'p']

dir_i = "dataset/all_train"
if not os.path.isdir(dir_i):
    os.mkdir(dir_i)
dir_l = "dataset/all_train_labels"
if not os.path.isdir(dir_l):
    os.mkdir(dir_l)

for d in datasets_names:

    dir = "dataset/" + d + "_train"
    files = glob.glob(dir+"/*.jpg")
    files = np.array(files)
    for filename in files:
        filename2 = '/'+filename.split('.')[0].split('/')[2]+d[0]+'.jpg'
        shutil.copy(filename, dir_i+filename2)

    dir = "dataset/" + d + "_train_labels"
    files = glob.glob(dir + "/*.png")
    for filename in files:
        if os.path.isfile(filename):
            filename2 = '/'+filename.split('.')[0].split('/')[2]+d[0]+ '.png'
            shutil.copy(filename, dir_l+filename2)