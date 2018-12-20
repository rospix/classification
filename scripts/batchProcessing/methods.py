import os
import csv
import numpy as np
import copy
import re

def loadImage(filename, sensor):

    image = np.zeros(shape=[256, 256])

    with open(filename, 'r') as csvfile:

        csvreader = csv.reader(csvfile, delimiter=' ')

        for row in csvreader:
            x = int(row[0])
            y = int(row[1])
            image.data[x, y] = row[2]

    return image

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def loadImages(sensor):

    path = "data/sensor_{}".format(sensor)

    file_names = os.listdir(path)

    file_names = sorted(file_names, key=numericalSort)

    images = []

    for i,filename in enumerate(file_names):
        images.append(loadImage(path+"/"+filename, sensor))

    return images
