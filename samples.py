import os
import glob
import shutil
import numpy as np

datasets_names = ['Butterfly',
                  'Giraffe',
                  'DogJump']
prop = 1


for d in datasets_names:

    dir = "dataset/" + d +"_train"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    files = glob.glob("../THUR15000/" + d + "/Src/*.jpg")
    files = np.array(files)
    np.random.shuffle(files)
    files = files[0:int(round(len(files)*prop))]
    print(' >> '+d+' train')
    print(str(len(files))+' total')

    for filename in files:
        shutil.copy(filename, dir)

    dir = "dataset/" + d + "_train_labels"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    files = glob.glob("../THUR15000/" + d + "/Src/*.png")
    print(str(len(files))+' labels')

    for filename in files:
        if os.path.isfile(filename):
            shutil.copy(filename, dir)

print(' >> Plane')
totalfiles = glob.glob("../THUR15000/Plane/Src/*.jpg")
print(str(len(totalfiles)) + ' total ')
labelsfiles = glob.glob("../THUR15000/Plane/Src/*.png")
print(str(len(labelsfiles)) + ' labels')

size_train = int(round(len(totalfiles)*0.5*prop))
print(str(size_train) + ' train ')
size_val = int(round(len(totalfiles)*0.25*prop))
print(str(size_val) + ' val ')
size_test = int(round(len(totalfiles)*0.25*prop))
print(str(size_test) + ' test ')

totalfiles = np.array(totalfiles)
np.random.shuffle(totalfiles)
train = totalfiles[0:size_train]
val = totalfiles[size_train:size_train+size_val]
test = totalfiles[size_train+size_val:size_train+size_val+size_test]

dirs = ["dataset/Plane_train",
        "dataset/Plane_val",
        "dataset/Plane_test"]

datasets = [train, val, test]

for i in range(0, 3):
    if not os.path.isdir(dirs[i]):
        os.mkdir(dirs[i])
        os.mkdir(dirs[i]+'_labels')

    for filename in datasets[i]:
        items = filename.split('/')
        item = items[len(items)-1].split('.')[0]
        shutil.copy(filename, dirs[i])
        file = "../THUR15000/Plane/Src/"+item+".png"
        if os.path.isfile(file):
            shutil.copy(file, dirs[i]+'_labels')

shutil.copy("../THUR15000/none.png", 'dataset')

