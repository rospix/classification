#!/usr/bin/env python

import rospy
import tf
import csv
import sys

import numpy

from dataProcessing.FeatureExtractor import FeatureExtractor
from dataProcessing.ImagePreprocessor import ImagePreprocessor

from rospix.msg import Image
from dataManager.DataContainer import BitmapContainer_Image, BitmapContainer_Segment, HistogramContainer

class Classification:

    def imageCallback(self, data):

        rospy.loginfo('getting images')

        bc = BitmapContainer_Image()

        np_image = numpy.zeros((256, 256))

        for i in range(0, 256):
            for j in range(0, 256):
                np_image[i, j] = data.image[i+255*j]

        segments = self.image_preprocessor.segmentate_image(np_image)
        features = self.feature_extractor.extract_features(segments)

        print("features: {}".format(features))

    def __init__(self):

        self.image = []

        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        rospy.init_node('rospix_classification', anonymous=True)

        # parameters
        # path = rospy.get_param('~path', '/')

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
