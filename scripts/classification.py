#!/usr/bin/env python2

import rospy
import tf
import csv
import sys
import numpy
import warnings

# from dataManager.DataContainer import BitmapContainer_Image, BitmapContainer_Segment
from dataManager.DataManager import POSSIBLE_LABELS, FEATURE_NAMES, NUMBERS_LABELS_MAP

# feature classification
from dataProcessing.FeatureExtractor import FeatureExtractor
from dataProcessing.ImagePreprocessor import ImagePreprocessor

# classifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

# ROS messages for images
from sensor_msgs.msg import Image
from rospix.msg import Image as RospixImage

# OpenCV + ROS cv_bridgfe
import cv2
from cv_bridge import CvBridge, CvBridgeError

class Classification:

    def imageCallback(self, data):

        # compy the image data to numpy matrix
        np_image = numpy.zeros((256, 256), numpy.uint16)
        for i,pixel in enumerate(data.image):
            x = i // 256
            y = i % 256
            np_image[x, y] = pixel
        np_image = np_image.astype(numpy.uint16)

        # segment the image
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

        # # print the segment lables
        # for i, track_type in enumerate(y_unknown):
        #     print("segment: {}, class: {}".format(i, NUMBERS_LABELS_MAP[track_type]))

        # equalize the image histogram
        hist,bins = numpy.histogram(np_image.flatten(),65536,[0,65535])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        cdf_m = numpy.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
        cdf = numpy.ma.filled(cdf_m,0).astype('uint16')
        img2 = cdf[np_image]

        # convert the image to 8bit range
        img2 = img2.astype(numpy.float)
        img_8bit = 255 - 255*img2/numpy.max(img2)
        img_8bit = img_8bit.astype(numpy.uint16)
        img_8bit = img_8bit.astype(numpy.uint8)

        # upscale the image to higher resolution
        upscale_factor = 8
        upscaled = cv2.resize(img_8bit, dsize=(256*upscale_factor, 256*upscale_factor), interpolation = cv2.INTER_AREA)

        # convert the image to colored format
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

        # highlight the segments
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

            cv2.rectangle(upscaled, (upscale_factor*(corners[0][1]),upscale_factor*(corners[0][0])+1), (upscale_factor*(corners[2][1]+1),upscale_factor*(corners[2][0]+1)), color, 3)

        # convert the image to ROS image
        image_message = self.bridge.cv2_to_imgmsg(upscaled, encoding="rgb8")

        # publish the image
        self.publisher_image.publish(image_message)

        # proba = -1*numpy.ones((y_unknown.shape[0], len(POSSIBLE_LABELS[1:])))
        # print("proba: {}".format(proba))

    def __init__(self):

        rospy.init_node('rospix_classification', anonymous=True)

        # initialize the ROS cv_bridge
        self.bridge = CvBridge()

        # parameters
        self.pipeline_file = rospy.get_param('~pipeline_path', '/')

        # load the sklearn pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.pipeline = joblib.load(self.pipeline_file) 

        # initialize the feature extraction
        self.feature_extractor = FeatureExtractor()
        self.image_preprocessor = ImagePreprocessor()

        # subscribers
        rospy.Subscriber("~image_in", RospixImage, self.imageCallback)

        # publishers
        self.publisher_image = rospy.Publisher("~classified_out", Image, queue_size=1)

        rospy.spin()

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
