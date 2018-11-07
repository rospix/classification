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
from dataManager.DataManager import POSSIBLE_LABELS, FEATURE_NAMES, NUMBERS_LABELS_MAP

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

NUMBERS_LABELS_MAP = {-1:'none',
                      1:'dot',
                      2:'blob_small',
                      3:'blob_big',
                      4:'blob_branched',
                      5:'track_straight',
                      6:'track_curly',
                      7:'drop',
                      8:'other',
                      9:'track_lowres'}

class Classification:

    def imageCallback(self, data):

        np_image = numpy.zeros((256, 256))

        for i in range(0, 256):
            for j in range(0, 256):
                np_image[i, j] = data.image[i+255*j]

        # segment the image
        segments = self.image_preprocessor.segmentate_image(np_image)

        if len(segments) == 0:
            return

        # prepare the numpy matrix for the (segments x features)
        np_features = numpy.zeros((len(segments), len(FEATURE_NAMES)))

        for i,segment in enumerate(segments):

            # extract the features from the image
            try:
                features = self.feature_extractor.extract_features_direct(segment)
            except:
                pass

            # for all feature names, extract copy the features in the correct order
            for j,feature_name in enumerate(FEATURE_NAMES):

                np_features[i, j] = features[feature_name]

        y_unknown = self.pipeline.predict(np_features)

        for i, track_type in enumerate(y_unknown):
            print("segment: {}, class: {}".format(i, NUMBERS_LABELS_MAP[track_type]))

        # proba = -1*numpy.ones((y_unknown.shape[0], len(POSSIBLE_LABELS[1:])))
        # print("proba: {}".format(proba))

    def __init__(self):

        rospy.init_node('rospix_classification', anonymous=True)

        self.image = []

        # parameters
        self.pipeline_file = rospy.get_param('~pipeline_path', '/')

        self.pipeline = joblib.load(self.pipeline_file) 

        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        # subscribers
        rospy.Subscriber("~image_in", Image, self.imageCallback)

        rate = rospy.Rate(10)

        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():

            rate.sleep()

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
