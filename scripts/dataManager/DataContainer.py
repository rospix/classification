import numpy as np

class BitmapContainer:
    '''
    Superclass for all bitmap containers. It specifies an interface which allows to store and
    retrieve bitmap and also its metadata. The metadata are not hardcoded into the class. They
    are stored in a dictitionary (self.metadata) and can be modified-added or removed-during
    runtime. Nevertheless, each child class (like BitmapContainer_Image and BitapContainer_Segment)
    have some predefined metadata-keys in this dictionary which are needed for the program
    to work.
    '''
    def __init__(self):
        self.bitmap = np.array([[0]])
        self.metadata = dict()

    def set_bitmap(self, bitmap):
        '''
        Stores bitmap in the container.
        :param bitmap: numpy array of arbitrary size (n_cols x n_rows)
        :return:
        '''
        self.bitmap = bitmap

    def get_bitmap(self):
        '''
        Retrieves bitmap from the container.
        :return:
        '''
        return self.bitmap

    def set_metadata(self, key, data):
        '''
        Set metadata value. Raises error when the key is not present in the dictionary.
        :param key: key to the metadata dictionary
        :param data: data which belongs to the key
        :return:
        '''
        keys = self.metadata.keys()
        if key in keys:
            self.metadata[key] = data
        else:
            raise ValueError('BitmapContainer key unknown. '
                             'You can add new key with add_metadata_key().')

    def get_metadata(self, key):
        '''
        Return value of metadata.
        :param key: key to the metadata dictionary
        :return: value belonging to the key
        '''
        if key == 'all':
            return self.metadata
        else:
            return self.metadata[key]

    def get_metadata_keys(self):
        '''
        :return: all keys of the self.metadata dictionary
        '''
        return self.metadata.keys()

    def add_metadata_key(self, key):
        '''
        Adds new metadata key and initializes the metadata entry with a value -1.
        :param key:
        :return:
        '''
        self.metadata[key] = -1


class BitmapContainer_Image(BitmapContainer):
    def __init__(self):
        self.bitmap = np.array([[0]])
        self.metadata = dict()
        self.add_metadata_key('id')
        self.add_metadata_key('name')


class BitmapContainer_Segment(BitmapContainer):
    def __init__(self):
        self.bitmap = np.array([[0]])
        self.metadata = dict()
        self.add_metadata_key('id')
        self.add_metadata_key('parent_im_id')
        self.add_metadata_key('corners')
        self.add_metadata_key('lbl_rough_shape_desired')
        self.add_metadata_key('lbl_shape_character_desired')
        self.add_metadata_key('lbl_rough_shape_classified')
        self.add_metadata_key('lbl_shape_character_classified')
        self.add_metadata_key('label_shape_classified')
        self.add_metadata_key('label_probs')
        self.add_metadata_key('label_shape_desired')
        self.add_metadata_key('features_dict')


class HistogramContainer():
    '''
    Container for histograms. It has got the same interface and behavior as the BitmapContainer, but
    the histogram is manipulated with methods set_histogram/get_histogram.
    '''
    def __init__(self):
        self.histogram = np.array([[0]])
        self.metadata = dict()

    def set_histogram(self, histogram):
        self.histogram = histogram

    def get_histogram(self):
        return self.histogram

    def set_metadata(self, key, data):
        keys = self.metadata.keys()
        if key in keys:
            self.metadata[key] = data
        else:
            raise ValueError('HistogramContainer key unknown. '
                             'You can add new key with add_metadata_key().')

    def get_metadata(self, key):
        return self.metadata[key]

    def get_metadata_keys(self):
        return self.metadata.keys()

    def add_metadata_key(self, key):
        self.metadata[key] = -1


class SumsContainer():
    '''
    Container for storage of row and column sums. The interface and behavior is the same
    as in the case of the BitmapContainer. The only difference are methods set_sums and get_sums,
    which accepts/returns two entries (lists of row and column sums).
    '''
    def __init__(self):
        self.sums_row = np.array([[0]])
        self.sums_column = np.array([[0]])
        self.metadata = dict()

    def set_sums(self, sums_row, sums_column):
        self.sums_row = sums_row
        self.sums_column = sums_column

    def get_sums(self):
        return self.sums_row, self.sums_column

    def set_metadata(self, key, data):
        keys = self.metadata.keys()
        if key in keys:
            self.metadata[key] = data
        else:
            raise ValueError('SumsContainer key unknown. '
                             'You can add new key with add_metadata_key().')

    def get_metadata(self, key):
        return self.metadata[key]

    def get_metadata_keys(self):
        return self.metadata.keys()

    def add_metadata_key(self, key):
        self.metadata[key] = -1
