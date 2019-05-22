# coding:utf-8
import os
import csv


def create_csv(dirname):
    path = './dataset/' + dirname + '/'

    name = os.listdir(path)

    with open('csv/'+dirname + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['data', 'label'])
        for n in name:
            if n[-4:] == '.jpg':

                file = './dataset/' + str(dirname) + '/' + str(n)
                labelfile = 'dataset/' + str(dirname) + '_labels/' + str(n[:-4] + '.png')
                if os.path.isfile(labelfile):
                    writer.writerow([file, './' + labelfile])
                else:
                    writer.writerow([file, './dataset/none.png'])
            else:
                pass


if __name__ == "__main__":
    create_csv('Butterfly_train')
    create_csv('DogJump_train')
    create_csv('Giraffe_train')
    create_csv('Plane_train')
    create_csv('Plane_val')
    create_csv('Plane_test')