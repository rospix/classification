#!/usr/bin/env python

import rospy
import tf
import csv
import sys

import numpy

from dataProcessing.FeatureExtractor import FeatureExtractor
from dataProcessing.ImagePreprocessor import ImagePreprocessor

from rospix.msg import Image
from dataManager.DataContainer import BitmapContainer_Image, BitmapContainer_Segment
from dataManager.DataManager import POSSIBLE_LABELS, FEATURE_NAMES

import pickle as pkl

from sklearn.pipeline import Pipeline

class Classification:

    def imageCallback(self, data):

        rospy.loginfo('getting images')

        bc = BitmapContainer_Image()

        np_image = numpy.zeros((256, 256))

        for i in range(0, 256):
            for j in range(0, 256):
                np_image[i, j] = data.image[i+255*j]

        segments = self.image_preprocessor.segmentate_image(np_image)

        for i,segment in enumerate(segments):

            features = self.feature_extractor.extract_features_direct(segment)

            np_features = numpy.zeros((1, len(features)))

            for i,feature in enumerate(features):

                np_features[0, i] = features[feature]
                # print("feature: {}, data: {}".format(feature, features[feature]))

            # if i == 0:

                # print("segment: {}".format(segment))
                # print("features: {}".format(features))

            # print("np_features: {}".format(np_features))

            y_unknown = self.pipeline.predict(np_features)

            proba = -1*numpy.ones((y_unknown.shape[0], len(POSSIBLE_LABELS[1:])))

            print("proba: {}".format(proba))
            print("y_unknown: {}".format(y_unknown))

    def __init__(self):

        rospy.init_node('rospix_classification', anonymous=True)

        self.image = []

        # parameters
        self.pipeline_file = rospy.get_param('~pipeline_path', '/')

        self.pipeline = pkl.load(open(self.pipeline_file, 'rb'))

        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        # subscribers
        rospy.Subscriber("~image_in", Image, self.imageCallback)

        rate = rospy.Rate(10)

        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():

            rospy.loginfo_throttle(1.0, 'Spinning')

            rate.sleep()

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
