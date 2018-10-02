'''
The source file for the class ImagePreprocessor, which contains routines for
a processing of whole images. The class can perform segmentation and also
simulates on-board image processing.
'''

from skimage import measure
from dataManager.DataContainer import BitmapContainer_Segment, \
    HistogramContainer, SumsContainer, BitmapContainer_Image
import numpy as np


class ImagePreprocessor():
    def __init__(self, A='', B='', C='', T=''):
        '''
        Constructor. Arguments A, B, C, D are matrices with Timepix calibration coefficients. These
        can be ommited when the calibration is not applied (default setting). If the calibration
        should be performed, it is necessary to supply valid implementation of a method self.calibrate.
        :param A:
        :param B:
        :param C:
        :param T:
        '''
        if len(A) > 0 and len(B) > 0 and len(C) > 0 and len(T) > 0:
            self.A = A
            self.B = B
            self.C = C
            self.T = T

    def pipeline(self, images_collection, calibrate=False):
        """
        Accepts list-like collection of BitmapContainer objects and performs image preprocessing
        :param images_collection:list-like collection of BitmapContainer instances.
        :return: list-like collection of segments (each segments represented as BitmapContainer_Segment)
        """
        all_segments = list()
        for i,image_dataframe in enumerate(images_collection):
            bitmap = image_dataframe.get_bitmap()
            if calibrate:
                bitmap = self.calibrate(bitmap, self.A, self.B, self.C, self.T)
            segments = self.segmentate_image(bitmap)
            for segment in segments:
                segment.set_metadata(key='parent_im_id',
                                     data=image_dataframe.get_metadata(key='id'))
                all_segments.append(segment)
            #print "Image ", i+1,"/",len(images_collection), "processed."
        return all_segments


    def segmentate_image(self, image):
        """
        :param image:
        :return: list of segments found in image
        """
        # threshold
        binary_mask = image > 0
        # binary analysis
        mask = measure.label(binary_mask)
        # generate segments
        segments = self.minimal_rectangular_hulls(mask, image)
        return segments


    def minimal_rectangular_hulls(self, image, original):
        '''
        Crops each unique labeled shape in image and returns list-like collection of
        BitmapContainer_Segment objects
        :param image: labeled image
        :return: 4 points of rectangle from upper left counterclockwise, format: (row, col)
        '''
        labels = np.unique(image[image.nonzero()])
        if np.max(labels) > 255:
            image = image.astype(dtype='uint16')
        else:
            image = image.astype(dtype='uint8')
        original = original.astype(dtype='uint8')
        coords = dict()
        for label in labels:
            coords[label] = list()
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row, col] != 0:
                    coords[image[row, col]].append((row, col))
        # Find corners for each label
        segments = list()
        for label in labels:
            pts = np.array(coords[label])
            # upper left: min row, min col
            up_left = (np.min(pts[:, 0]), np.min(pts[:, 1]))
            # lower left: max row, min col
            low_left = (np.max(pts[:, 0]), np.min(pts[:, 1]))
            # lower right: max row, max col
            low_right = (np.max(pts[:, 0]), np.max(pts[:, 1]))
            # upper right: min row, max col
            up_right = (np.min(pts[:, 0]), np.max(pts[:, 1]))
            if (up_left[0] == 0
                and up_left[1] == 0
                and low_right[0] == image.shape[0] - 1
                and low_right[1] == image.shape[1] - 1):
                continue
            else:
                corners = np.array([up_left, low_left, low_right, up_right])
                if (np.unique(corners, axis=0)).shape[0] == 1:
                    is_one_point = True
                else:
                    is_one_point = False
                if is_one_point:
                    image_data = np.atleast_2d(original.copy()[corners[0, 0], corners[0, 1]])
                    mask = np.atleast_2d(image.copy()[corners[0, 0], corners[0, 1]])
                else:
                    image_data = np.atleast_2d(original.copy()[corners[0, 0]:corners[2, 0] + 1,
                                                     corners[0, 1]:corners[2, 1] + 1])
                    mask = np.atleast_2d(image.copy()[corners[0, 0]:corners[2, 0] + 1,
                                                     corners[0, 1]:corners[2, 1] + 1])

                for row in range(image_data.shape[0]):
                    for col in range(image_data.shape[1]):
                        if int(mask[row][col]) != int(label):
                            image_data[row][col] = 0

                bc = BitmapContainer_Segment()
                bc.set_bitmap(image_data)
                bc.set_metadata(key='corners', data=corners)
                segments.append(bc)
        return segments


    def calibrate(self, image, A, B, C, T):
        '''
        Calibrate one image. Currently not implemented since it was not needed in
        the thesis.
        :param image: 2D numpy array
        :param A:
        :param B:
        :param C:
        :param T:
        :return: 2D numpy array
        '''
        print "ImagePreprocessor.calibrate(): Not implemented"
        #return (T*image+B-C)/(image-A)
        return image


    def generate_calibrated(self, images_collection, A, B, C, T):
        '''
        Calibrate sequence of images. Currently not implemented since it was not needed in
        the thesis.
        :param images_collection: list of BitmapContainer_Image instances
        :param A:
        :param B:
        :param C:
        :param T:
        :return: list of BitmapContainer_Image instances with calibrated images
        '''
        calibrated = list()
        for image_container in images_collection:
            image = image_container.get_bitmap()
            image = self.calibrate(image, A, B, C, T)
            ic = BitmapContainer_Image()
            ic.set_bitmap(image)
            calibrated.append(ic)
        return calibrated


    def generate_histograms(self, images_collection, bins=np.linspace(0,140,17)):
        """
        Generate sequence of energetic histograms from sequence of images.
        :param images_collection: list of BitmapContainer_Image instances
        :param bins:
        :return: list of HistogramContainer objects
        """
        hists = list()
        for image_cont in images_collection:
            image = image_cont.get_bitmap()
            hist,b = np.histogram(image[image>0].flatten(), bins=bins)
            h = HistogramContainer()
            h.set_histogram(hist)
            hists.append(h)
        return hists

    def generate_sums(self, images_collection):
        """
        Generate row and column sums for sequence of images.
        :param images_collection: list of BitmapContainer_Image instances
        :return: list of SumsContainer objects
        """
        sums = list()
        for image_container in images_collection:
            image = image_container.get_bitmap().copy()
            image[image > 0] = 1
            sum_row = np.sum(image, axis=0)
            sum_col = np.sum(image, axis=1)
            s = SumsContainer()
            s.set_sums(sums_row=sum_row, sums_column=sum_col)
            sums.append(s)
        return sums

    def generate_binning(self, images_collection, binning_mode='binning16'):
        """
        Perform binning on a sequence of images.
        :param images_collection: list of BitmapContainer_image instances
        :param binning_mode:
        :return: list ob BitmapContainer_image instances containing binned images
        """
        binnings = list()
        for image_container in images_collection:
            image = image_container.get_bitmap()
            binned  = self.binning(image, binning_mode=binning_mode)
            ic = BitmapContainer_Image()
            ic.set_bitmap(binned)
            binnings.append(ic)
        return binnings

    def binning(self, image, binning_mode='binning16'):
        """
        Perform binning in one image.
        :param image: 2D numpy array
        :param binning_mode:
        :return: binned image in 2D numpy array
        """
        if binning_mode == 'binning8':
            binSize = 32
        elif binning_mode == 'binning16':
            binSize = 16
        elif binning_mode == 'binning32':
            binSize = 8
        else:
            print('binning(): unknown binning mode. Valid options: binning8, binning16, binning32.')
        nrBins = image.shape[0] / binSize
        binned = np.zeros((nrBins, nrBins))
        for i in range(nrBins):
            for j in range(nrBins):
                binned[i, j] = np.count_nonzero(image[binSize * i:binSize * i + binSize,
                                                binSize * j:binSize * j + binSize])
        return binned
