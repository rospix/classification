#!/usr/bin/env python

import rospy
import tf
import csv
import sys

import numpy

from dataProcessing.FeatureExtractor import FeatureExtractor
from dataProcessing.ImagePreprocessor import ImagePreprocessor

from rospix.msg import Image as RospixImage
from dataManager.DataContainer import BitmapContainer_Image, BitmapContainer_Segment
from dataManager.DataManager import POSSIBLE_LABELS, FEATURE_NAMES, NUMBERS_LABELS_MAP

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

from ppretty import ppretty

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

        np_image = numpy.zeros((256, 256), numpy.uint16)

        for i,pixel in enumerate(data.image):

            x = i // 256
            y = i % 256

            np_image[x, y] = pixel

        np_image = np_image.astype(numpy.uint16)

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

        # for i, track_type in enumerate(y_unknown):
        #     print("segment: {}, class: {}".format(i, NUMBERS_LABELS_MAP[track_type]))

        hist,bins = numpy.histogram(np_image.flatten(),65536,[0,65535])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        cdf_m = numpy.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = numpy.ma.filled(cdf_m,0).astype('uint16')
        img2 = cdf[np_image]

        img2 = img2.astype(numpy.float)
        img_8bit = 255 - 255*img2/numpy.max(img2)
        img_8bit = img_8bit.astype(numpy.uint16)
        img_8bit = img_8bit.astype(numpy.uint8)

        scale_factor = 8
        upscaled = cv2.resize(img_8bit, dsize=(256*scale_factor, 256*scale_factor), interpolation = cv2.INTER_AREA)

        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

        for i, segment in enumerate(segments):

            corners = segment.get_metadata(key="corners")

            if (y_unknown[i] >= 2 and y_unknown[i] <= 4) or (y_unknown[i] == 7): # protons/alphas
                color = (255, 0, 0)
            elif (y_unknown[i] >= 5 and y_unknown[i] <= 6) or (y_unknown[i] == 9): # electrons
                color = (255, 255, 0)
            elif y_unknown[i] == 1: # photons
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(upscaled, (scale_factor*(corners[0][1]),scale_factor*(corners[0][0])+1), (scale_factor*(corners[2][1]+1),scale_factor*(corners[2][0]+1)), color, 3)

        image_message = self.bridge.cv2_to_imgmsg(upscaled, encoding="rgb8")

        self.publisher_image.publish(image_message)

        # proba = -1*numpy.ones((y_unknown.shape[0], len(POSSIBLE_LABELS[1:])))
        # print("proba: {}".format(proba))

    def __init__(self):

        rospy.init_node('rospix_classification', anonymous=True)

        self.bridge = CvBridge()

        self.image = []

        # parameters
        self.pipeline_file = rospy.get_param('~pipeline_path', '/')

        self.pipeline = joblib.load(self.pipeline_file) 

        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        # subscribers
        rospy.Subscriber("~image_in", RospixImage, self.imageCallback)

        self.publisher_image = rospy.Publisher("image_out", Image)

        rate = rospy.Rate(10)

        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():

            rate.sleep()

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
