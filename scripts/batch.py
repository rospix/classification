#!/usr/bin/env python2

import os
import rospy
import tf
import csv
import sys
import numpy
import warnings
import re

# from dataManager.DataContainer import BitmapContainer_Image, BitmapContainer_Segment
from dataManager.DataManager import POSSIBLE_LABELS, FEATURE_NAMES, NUMBERS_LABELS_MAP

# feature classification
from dataProcessing.FeatureExtractor import FeatureExtractor
from dataProcessing.ImagePreprocessor import ImagePreprocessor

# classifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from rospix_classification.msg import Pixel
from rospix_classification.msg import Cluster
from rospix_classification.msg import ProcessedImage

class Classification:

    # #{ loadImage()
    
    def loadImage(self, filename):
    
        image = numpy.zeros(shape=[256, 256])
    
        with open(filename, 'r') as csvfile:
    
            csvreader = csv.reader(csvfile, delimiter=' ')
    
            for i,row in enumerate(csvreader):
    
                for j in range(0, 256):
    
                    image[i, int(j)] = row[j]
    
        return image
    
    # #} end of loadImage()

   # #{ numericalSort()
   
    def numericalSort(self, value):

        parts = self.numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
    
   
   # #} end of numericalSort()    return parts

    def __init__(self):

        rospy.init_node('rospix_classification', anonymous=True)

        self.numbers = re.compile(r'(\d+)')

        # parameters
        self.pipeline_file = rospy.get_param('~pipeline_path', '/')
        self.source_path = rospy.get_param('~source_path', '/')
        self.result_path = rospy.get_param('~result_path', '/')
        self.delimiter = rospy.get_param('~delimiter', '/')
        self.line_ending = rospy.get_param('~line_ending', '/')

        # load the sklearn pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.pipeline = joblib.load(self.pipeline_file)

        # initialize the feature extraction
        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        # load the files
        file_names = os.listdir(self.source_path)
        file_names = sorted(file_names, key=self.numericalSort)

        # for all images
        for i,filename in enumerate(file_names):

            rospy.loginfo('processing image {}'.format(filename))

            np_image = self.loadImage(self.source_path+"/"+filename)
            np_image = np_image.astype(numpy.uint16)

            if numpy.count_nonzero(np_image) > 0:
                image_empty = False
            else:
                image_empty = True

            y_unknown = []

            # segment the image
            if not image_empty:

                segments = self.image_preprocessor.segmentate_image(np_image)

                # prepare the numpy matrix for the (segments x features)
                np_features = numpy.zeros((len(segments), len(FEATURE_NAMES)))

                # for all localized segments
                for i,segment in enumerate(segments):

                    # extract the features from the image
                    try:
                        features = self.feature_extractor.extract_features_direct(segment)
                    except:
                        pass

                    # for all feature names, extract copy the features in the correct order
                    for j,feature_name in enumerate(FEATURE_NAMES):

                        np_features[i, j] = features[feature_name]

                # classify the segments into classes
                y_unknown = self.pipeline.predict(np_features)

            processed_data = ProcessedImage()

            # create the output file and the image cluester statistics
            with open(self.result_path+"/"+filename+".statistics.txt", "w") as outfile:

                for i in range(0, 10):
                    outfile.write("{}: {}{}".format(POSSIBLE_LABELS[i], numpy.count_nonzero(y_unknown == i), self.line_ending))

            # create the output file and the cluster list
            with open(self.result_path+"/"+filename+".clusters.txt", "w") as outfile:

                # if the image is not empty, fill the "data" msg
                if not image_empty:

                    # for all localized segments
                    for i,segment in enumerate(segments):

                        new_cluster = Cluster()

                        # for all pixels in the image
                        for j in range(0, segment.bitmap.shape[0]):
                            for k in range(0, segment.bitmap.shape[1]):

                                # if the pixel is nonzero
                                if segment.bitmap[j, k] > 0:

                                    # copy the pixel in the message
                                    new_pixel = Pixel()
                                    new_pixel.x = j
                                    new_pixel.y = k
                                    new_pixel.value = segment.bitmap[j, k]

                                    new_cluster.pixels.append(new_pixel)

                        # copy the corner coordinate (left-upper)
                        corners = segment.get_metadata(key="corners")
                        new_cluster.lu_x = corners[0, 0]
                        new_cluster.lu_y = corners[0, 1]

                        # copy the cluster cluster class and name
                        new_cluster.cluster_class.cluster_class = y_unknown[i]
                        new_cluster.cluster_class.name = POSSIBLE_LABELS[y_unknown[i]]

                        processed_data.cluster_list.append(new_cluster);

                proba = -1*numpy.ones((y_unknown.shape[0], len(POSSIBLE_LABELS[1:])))

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
