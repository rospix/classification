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

# ROS messages for publishing
from rospix_classification.msg import Pixel
from rospix_classification.msg import Cluster
from rospix_classification.msg import ProcessedImage

# OpenCV + ROS cv_bridge
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

        if numpy.count_nonzero(np_image) > 0:
            image_empty = False
        else:
            image_empty = True

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
        if image_empty:
            img_8bit = 255 - 255*img2/1.0
        else:
            img_8bit = 255 - 255*img2/numpy.max(img2)
        img_8bit = img_8bit.astype(numpy.uint16)
        img_8bit = img_8bit.astype(numpy.uint8)

        # upscale the image to higher resolution
        upscale_factor = 8
        upscaled = cv2.resize(img_8bit, dsize=(256*upscale_factor, 256*upscale_factor), interpolation = cv2.INTER_AREA)

        # convert the image to colored format
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

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

            # # print the segment lables
            # for i, track_type in enumerate(y_unknown):
            #     print("segment: {}, class: {}".format(i, NUMBERS_LABELS_MAP[track_type]))

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

        # publish the processed data

        processed_data = ProcessedImage()

        # if the image is not empty, fill the "data" msg
        if not image_empty:

            # the number of counts of the track types
            processed_data.cluster_counts.dot = numpy.count_nonzero(y_unknown == 1)
            processed_data.cluster_counts.blob_small = numpy.count_nonzero(y_unknown == 2)
            processed_data.cluster_counts.blob_big = numpy.count_nonzero(y_unknown == 3)
            processed_data.cluster_counts.blob_branched = numpy.count_nonzero(y_unknown == 4)
            processed_data.cluster_counts.track_straight = numpy.count_nonzero(y_unknown == 5)
            processed_data.cluster_counts.track_curly = numpy.count_nonzero(y_unknown == 6)
            processed_data.cluster_counts.drop = numpy.count_nonzero(y_unknown == 7)
            processed_data.cluster_counts.other = numpy.count_nonzero(y_unknown == 8)
            processed_data.cluster_counts.track_lowres = numpy.count_nonzero(y_unknown == 9)

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

        self.publisher_processed.publish(processed_data)

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
        rospy.Subscriber("~image_in", RospixImage, self.imageCallback, queue_size=1)

        # publishers
        self.publisher_image = rospy.Publisher("~labeled_out", Image, queue_size=1)

        # publisher for the classification metadata
        self.publisher_processed = rospy.Publisher("~data", ProcessedImage, queue_size=1)

        rospy.spin()

if __name__ == '__main__':
    try:
        classification = Classification()
    except rospy.ROSInterruptException:
        pass
