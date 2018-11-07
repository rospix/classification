# -*- coding: 'utf-8' -*-

import os.path
import pickle as pkl
import sqlite3 as sq
from os import listdir
import numpy as np

from DataContainer import BitmapContainer_Image, BitmapContainer_Segment, HistogramContainer

FEATURE_NAMES = ['area', 'energy', 'nr_crossings', 'chull_area', 'pixel_energy',
                 'chull_occupancy', 'linear_fit', 'skeleton_chull_ratio',
                 'pixels_direction', 'tortuosity', 'eig_ratio', 'dist_mean', 'dist_std',
                 'boxiness', 
                 'diagonaliness', 'straightness', 
                 'q10', 'q50', 'q90',
                 'e1', 'e2', 'e3', 'e4',
                 'bheigth', 'bwidth']

FEATURE_TYPES = ['INT', 'FLOAT', 'INT', 'INT', 'FLOAT', 'FLOAT',
                 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 
                 'INT',
                 'INT','INT', 
                 'FLOAT', 'FLOAT', 'FLOAT', 
                 'INT', 'INT', 'INT', 'INT',
                 'INT', 'INT']


LABELS_NUMBERS_MAP = {'none':-1,
                      'dot':1,
                      'blob_small':2,
                      'blob_big':3,
                      'blob_branched':4,
                      'track_straight':5,
                      'track_curly':6,
                      'drop':7,
                      'other':8,
                      'track_lowres':9}

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

# First label must be none.
POSSIBLE_LABELS = ['none',
                   'dot',
                   'blob_small',
                   'blob_big',
                   'blob_branched',
                   'track_straight',
                   'track_curly',
                   'drop',
                   'other',
                   'track_lowres']


class DataManager_sql_2():
    def __init__(self, db_name, im_folder):
        '''
        This function instantiates DataManager object and connects it with the SQL database. If the database does not
        exist, empty one will be created.
        :param db_name: database file
        :param im_folder: absolute path to folder with images
        :return DataManager instance
        '''

        if os.path.isfile(db_name) == False:
            self.db = sq.connect(db_name)
            cursor = self.db.cursor()
            print("DataManager: database file not detected. Creating new database...")
            cursor.execute("CREATE TABLE tbl_images(id_im INTEGER PRIMARY KEY, "
                           "path TEXT, "
                           "bias INT, "
                           "chunk_id INT, "
                           "exposure INT, "
                           "favorite INT, "
                           "filtered_pixels INT, "
                           "filtering INT, "
                           "got_metadata INT, "
                           "hidden INT, "
                           "vzlusat_id INT, "
                           "max_filtered INT, "
                           "max_original INT, "
                           "min_filtered INT, "
                           "min_original INT, "
                           "mode INT, "
                           "original_pixels INT, "
                           "latitude FLOAT, "
                           "longitude FLOAT, "
                           "pxl_limit INT, "
                           "temperature INT, "
                           "time INT, "
                           "type INT, "
                           "uv1_thr INT,"
                           "tag INT)")

            clf_probs = ''
            for i,_ in enumerate(POSSIBLE_LABELS):
                clf_probs = clf_probs+('p_cl_'+str(i)+' FLOAT, ')
            feat_type = ''
            for i,_ in enumerate(FEATURE_NAMES):
                feat_type = feat_type+FEATURE_NAMES[i]+' '+FEATURE_TYPES[i]+','
            query = ["CREATE TABLE tbl_segments("
                           "id_seg INTEGER PRIMARY KEY,"
                           "id_im INTEGER,"
                           "label_shape_desired INT,"
                           "label_shape_classified INT,"+clf_probs+
                           "coor_lu_x INT,"
                           "coor_lu_y INT,"
                           "coor_rl_x INT,"
                           "coor_rl_y INT,"+feat_type+
                           "FOREIGN KEY(id_im) REFERENCES tbl_images(id_im))"]
            cursor.execute(query[0])
            print("DataManager: database ready.")
            self.db.commit()
        else:
            print("DataManager: database found. Connecting...")
            self.db = sq.connect(db_name)
            cursor = self.db.cursor()
            print("DataManager: database ready.")
        self.cursor = cursor
        self.folder = im_folder
        self.NUMBER_TO_LABEL = NUMBERS_LABELS_MAP
        self.LABEL_TO_NUMBER = LABELS_NUMBERS_MAP

    def summary(self):
        '''
        :return: string with information about database
        '''
        s = '****** DB summary ******\n' \
            'Nr images: ' + str(self.cursor.execute('SELECT COUNT(*) FROM tbl_images').fetchone()[0])+ ' \n' \
            'Nr segments: ' + str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments').fetchone()[0]) + '\n' \
            'Nr labeled (dot): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 1').fetchone()[0]) + \
            '\n' \
            'Nr labeled (blob_small): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 2').fetchone()[0]) + \
            '\n' \
            'Nr labeled (blob_big): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 3').fetchone()[0]) + \
            '\n' \
            'Nr labeled (blob_branched): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 4').fetchone()[0]) + \
            '\n' \
            'Nr labeled (track_curly): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 5').fetchone()[0]) + \
            '\n' \
            'Nr labeled (track_straight): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 6').fetchone()[0]) + \
            '\n''Nr labeled (drop): ' + \
            str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 7').fetchone()[0]) + \
            '\n' + \
        'Nr labeled (other): ' + \
        str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 8').fetchone()[0]) + \
        '\n'      + \
        'Nr labeled (track_lowres): ' + \
        str(self.cursor.execute('SELECT COUNT(*) FROM tbl_segments WHERE label_shape_desired = 9').fetchone()[0]) + \
        '\n'         
        return s

    def get_image_ids(self, filter='fullres'):
        """
        :filter: string with more possibilities:
        'fullres': all images with tag=1, it means that image contains meaningful image data,
        'binning8', 'binning16', 'binning32', 'sums', 'histogram': get ids of dataframes created in a given mode,
        '<query>' query in format 'attribute_1=val_1 AND/OR attribute_2=val_2...'. This automatically returns
        noncorrupted ids which corresponds to the given query. For attribute names refer the constructor.
        :return: list of ids of all images (integers)
        """
        if filter == 'fullres':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=1").fetchall()
        elif filter == 'binning8':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=2").fetchall()
        elif filter == 'binning16':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=4").fetchall()
        elif filter == 'binning32':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=8").fetchall()
        elif filter == 'sums':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=16").fetchall()
        elif filter == 'histogram':
            ids_2D = self.cursor.execute("SELECT id_im FROM tbl_images WHERE tag=1 AND type=32").fetchall()
        else:
            query = "SELECT id_im FROM tbl_images WHERE tag=1 AND " + filter
            ids_2D = self.cursor.execute(query).fetchall()
        ids = [element for tupl in ids_2D for element in tupl]
        return ids
    
    def get_histograms(self, im_ids):
        '''
        Get histograms with specified ids.
        :param im_ids: ids of histograms to retrieve from the database
        :return: list of histograms contained within HistogramContainer class instances.
        '''
        # get vzlusat ids of frames
        histograms = list()
        mask = np.ones((len(im_ids), 1))
        for i,im_id in enumerate(im_ids):
            vzlusat_id = self.cursor.execute("SELECT vzlusat_id FROM tbl_images WHERE id_im=?", (im_id,)).fetchone()
            histogram_path = self.cursor.execute("SELECT path FROM tbl_images WHERE vzlusat_id=? AND type=32", vzlusat_id).fetchone()
            if histogram_path == None:
                mask[i] = -1
                continue
            else:   
                histogram_path = self.folder+'/'+histogram_path[0]
                hist = pkl.load(open(histogram_path, 'rb'))
                h = HistogramContainer()
                h.set_histogram(hist.data)
                histograms.append(h)
        return histograms, mask

    def get_ids_regression_training_pairs(self, type1='1', type2='8'):
        '''
        :param type1:
        :param type2:
        :return:
        '''
        # 1) Get vzlusat ids, which has got all variants of training images
        query1 = 'SELECT id_im, vzlusat_id, type FROM tbl_images WHERE vzlusat_id IN(' \
                ' SELECT vzlusat_id FROM tbl_images WHERE type='+type1+')' \
                ' AND vzlusat_id IN(SELECT vzlusat_id FROM tbl_images WHERE type='+type2+')' \
                ' AND type=' + type1 + \
                ' ORDER BY vzlusat_id'
        query2 = 'SELECT id_im, vzlusat_id, type FROM tbl_images WHERE vzlusat_id IN(' \
                 ' SELECT vzlusat_id FROM tbl_images WHERE type=' + type1 + ')' \
                 ' AND vzlusat_id IN(SELECT vzlusat_id FROM tbl_images WHERE type=' + type2 + ')' \
                 ' AND type=' + type2 + \
                 ' ORDER BY vzlusat_id'
        ids1_2D = self.cursor.execute(query1).fetchall()
        ids1 = [element for tupl in ids1_2D for element in tupl]
        ids2_2D = self.cursor.execute(query2).fetchall()
        ids2 = [element for tupl in ids2_2D for element in tupl]
        return ids1, ids2
    

    def get_custom_column(self, query):
        '''
        Get values of one column with fully custom SQL query. The query is directly forwarded to the SQL backend without control
        of possible side effects, so it is possible to corrupt the database with it. It should be used carefully.
        :param query:
        :return:
        '''
        result = self.cursor.execute(query)
        return result

    def get_segment_ids(self, filter='off', im_id = ''):
        """
        :param filter: string with following possibilities:
        'off': return all segments in the database,
        'desired': return all segments which are annotated by expert,
        'id_im': return all segments which are contained in image with id im_id (second paraeter of this function)
        '<query>': custom query in a form 'attribute1=1 AND/OR attribute2=2...'. Basically all SQL constructs can be
        used, the filter query is part of a standard SQL query which is located after the WHERE statement.
        :param im_id: used in conjunction with 'id_im' filtering.
        :return: list of segment ids
        """

        if filter == 'off':
            ids_2D = self.cursor.execute("SELECT id_seg FROM tbl_segments").fetchall()
            ids = [element for tupl in ids_2D for element in tupl]
            return ids
        if filter == 'desired':
            ids_2D = self.cursor.execute("SELECT id_seg FROM tbl_segments WHERE label_shape_desired <> -1").fetchall()
            ids = [element for tupl in ids_2D for element in tupl]
            return ids
        if filter == 'id_im':
            ids_2D = self.cursor.execute(
                "SELECT id_seg FROM tbl_segments WHERE id_im = ?",(im_id,)).fetchall()
            ids = [element for tupl in ids_2D for element in tupl]
            return ids
        if isinstance(filter, (int, long)):
            ids_2D = self.cursor.execute(
                "SELECT id_seg FROM tbl_segments WHERE id_im = ? AND label_shape_desired = ?",(im_id,filter)).fetchall()
            ids = [element for tupl in ids_2D for element in tupl]
            return ids
        else:
            query = "SELECT id_seg FROM tbl_segments WHERE " + filter
            #query = "SELECT id_seg FROM tbl_labels_desired WHERE rough_shape = %s" % ('"'+filter+'"')
            ids_2D = self.cursor.execute(query).fetchall()
            ids = [element for tupl in ids_2D for element in tupl]
            return ids

    def get_images(self, ids, filter='all'):
        '''
        Loads images with given ids.
        :param ids: ids of images to load from binary files
        :param filter: simple further filtering of images. Normally it is not needed since all the filtering can be done
        more efficiently in method get_image_ids().
        :return: list of Image objects
        '''
        l = tuple(ids)
        placeholder = '?'  # For SQLite. See DBAPI paramstyle.
        placeholders = ', '.join(placeholder for unused in l)
        if filter=='noncorrupted':
            query = 'SELECT path ' \
                    'FROM tbl_images WHERE (id_im IN (%s) AND tag = 1)' % placeholders
        else:
            query = 'SELECT path ' \
                    'FROM tbl_images WHERE id_im IN (%s)' % placeholders
        self.cursor.execute(query, l)
        paths = self.cursor.fetchall()
        paths = [element for tupl in paths for element in tupl]
        images = []
        for i,path in enumerate(paths):
            end = path.split('.')[-1]
            if end == 'pkl':
                try:
                    binary_im = pkl.load(open(self.folder+'/'+path, 'rb'))
                    bc = BitmapContainer_Image()
                    bc.set_bitmap(binary_im.data)
                    bc.set_metadata(key='id', data=ids[i])
                    bc.set_metadata(key='name', data=path)
                    images.append(bc)
                except IOError:
                    print 'DataManager.get_images: file ', self.folder+'/'+path, ' not found. Skipping.'
        return images

    def get_segment_corners(self, ids):
        """
        Returns bounding rectangles of segments.
        :param ids: ids of segments
        :return: list of numpy arrays of the size 4x2 which contains corners of bounding rectangles of queried segments.
        """
        ret = list()
        for id in ids:
            corners = list()
            query_lu_x = 'SELECT coor_lu_x FROM tbl_segments WHERE id_seg = %s' % str(id)
            query_lu_y = 'SELECT coor_lu_y FROM tbl_segments WHERE id_seg = %s' % str(id)
            query_rl_x = 'SELECT coor_rl_x FROM tbl_segments WHERE id_seg = %s' % str(id)
            query_rl_y = 'SELECT coor_rl_y FROM tbl_segments WHERE id_seg = %s' % str(id)
            self.cursor.execute(query_lu_x)
            corners.append(self.cursor.fetchone())
            self.cursor.execute(query_lu_y)
            corners.append(self.cursor.fetchone())
            self.cursor.execute(query_rl_x)
            corners.append(self.cursor.fetchone())
            self.cursor.execute(query_rl_y)
            corners.append(self.cursor.fetchone())

            corners_4 = np.array([[corners[0][0], corners[1][0]],
                                  [corners[2][0], corners[1][0]],
                                  [corners[2][0], corners[3][0]],
                                  [corners[0][0], corners[3][0]]])
            ret.append(corners_4)
        return ret

    def get_segments(self, ids):
        """
        Returns segments with specified IDs.
        :param ids: IDs of segments to return
        :return: list of BitmapContainer_Segment objects, each contains one segment.
        """
        placeholders = self.generate_placeholders(len(ids))
        # Open images and extract segments
        segments = list()
        for id_seg in ids:
            # Get ID of image of current segment and load this image
            query = 'SELECT id_im FROM tbl_segments WHERE id_seg='+str(id_seg)
            self.cursor.execute(query)
            id_im = self.cursor.fetchall()[0]
            image_container = self.get_images(id_im)[0]
            image = image_container.get_bitmap()

            # Get corners of border of current segment and crop loaded image
            query = 'SELECT ' \
                    'coor_lu_x, coor_lu_y, ' \
                    'coor_rl_x, coor_rl_y ' \
                    'FROM tbl_segments WHERE id_seg='+str(id_seg)
            self.cursor.execute(query)
            corners = self.cursor.fetchone()

            corners = np.array([[corners[0], corners[1]],
                                  [corners[2], corners[1]],
                                  [corners[2], corners[3]],
                                  [corners[0], corners[3]]])

            # get label_shape_desired
            query = 'SELECT label_shape_desired, label_shape_classified FROM tbl_segments WHERE id_seg='+str(id_seg)
            self.cursor.execute(query)
            label_desired, label_classified = self.cursor.fetchone()
            bc = BitmapContainer_Segment()
            if (np.unique(corners, axis=0)).shape[0] == 1:
                image_data = np.atleast_2d(image[corners[0, 0], corners[0, 1]])
            else:
                image_data = np.atleast_2d(image[corners[0, 0]:corners[2, 0] + 1,
                                           corners[0, 1]:corners[2, 1] + 1])
            feats = self.get_features((id_seg,))[0,:]
            features_dict = {}
            for i, val in enumerate(feats):
                features_dict[FEATURE_NAMES[i]] = val

            # get class probabilities
            get_names = ''
            for k, l in enumerate(POSSIBLE_LABELS[1:]):
                get_names = get_names + 'p_cl_' + str(k + 1) + ' ,'
            get_names = get_names[0:-1]
            query = 'SELECT '+get_names+ 'FROM tbl_segments WHERE id_seg=?'
            self.cursor.execute(query,(id_seg,))
            probs = np.array(self.cursor.fetchone())

            bc.set_metadata('features_dict', features_dict)
            bc.set_bitmap(image_data)
            bc.set_metadata(key='id', data=id_seg)
            bc.set_metadata(key='parent_im_id', data=image_container.get_metadata(key='id'))
            bc.set_metadata(key='label_shape_desired', data=label_desired)
            bc.set_metadata(key='label_shape_classified', data=label_classified)
            bc.set_metadata(key='label_probs', data=probs)
            bc.set_metadata(key='corners', data=corners)
            segments.append(bc)
        return segments

    def get_features(self, ids):
        """
        Get matrix with features ready to use with scikit-learn.
        :param ids: list of ints (ids of segments)
        :return: numpy array of features (nr_segments x nr_features), columns ordered as FEATURE_NAMES variable
        (see beginning of this source file).
        """
        np_features = np.zeros((len(ids), len(FEATURE_NAMES)))
        for k,ID in enumerate(ids):
            # find row with particular ID
            names = FEATURE_NAMES[0]
            for feat_name in FEATURE_NAMES[1:]:
                names = names + ',' + feat_name
            query = 'SELECT %s FROM tbl_segments WHERE id_seg=' % names +str(ID)
            self.cursor.execute(query)
            row = self.cursor.fetchall()[0]
            for j,col in enumerate(row):
                np_features[k,j] = row[j]
        return np_features

    def insert_new_segments(self, segments, check_duplicity=True):
        """
        Insert new segments into the database. Already existing segments will be updated so this routine can be safely
        run e.g. after feature computation to reflect changes in the feature extraction algorithms.
        :param segments: list of BitmapContainers_Segment
        :return: check_duplicity: only for special purposes. It should be left on True.
        """

        # Make list of all present segment ids and segments not present
        # if segment exists, its id should be in database
        segments_to_update = list()
        segments_to_insert = list()
        ids_to_update = list()
        for i,segment in enumerate(segments):
            if check_duplicity == True:
                # check if segment is not present
                query = "SELECT id_seg FROM tbl_segments WHERE (" \
                        "id_im = ? AND " \
                        "coor_lu_x = ? AND " \
                        "coor_lu_y = ? AND " \
                        "coor_rl_x = ? AND " \
                        "coor_rl_y = ?) LIMIT 1"
                corners = segment.get_metadata('corners')
                self.cursor.execute(query, (segment.get_metadata('parent_im_id'),
                                            int(corners[0,0]),
                                            int(corners[0,1]),
                                            int(corners[2,0]),
                                            int(corners[2,1])))
                data = self.cursor.fetchone()
                if data is not None:
                    segments_to_update.append(segment)
                    ids_to_update.append(data[0])
                else:
                    segments_to_insert.append(segment)
            else:
                segments_to_insert.append(segment)

        # Now add all segments on a list segments_to_insert
        pl = self.generate_placeholders(nr_placeholders=7+len(FEATURE_NAMES))
        names = self.generate_seg_props(FEATURE_NAMES)
        for i, segment in enumerate(segments_to_insert):
            features = [segment.get_metadata('features_dict')[key] for key in FEATURE_NAMES]
            corners = segment.get_metadata('corners')

            query_values = [segment.get_metadata('parent_im_id'),
                            int(corners[0, 0]),
                            int(corners[0, 1]),
                            int(corners[2, 0]),
                            int(corners[2, 1]),
                           segment.get_metadata('label_shape_desired'),
                           segment.get_metadata('label_shape_classified'),
                           ]
            query_values.extend(features)
            query = 'INSERT INTO tbl_segments(%s) VALUES(%s)' % (names, pl)
            self.cursor.execute(query, query_values)
        self.db.commit()
        # Update all segments on a list segments_to_update
        # !! we are updating only features...
        pl = self.generate_placeholders(nr_placeholders=len(FEATURE_NAMES))
        set_string = ''
        for f in FEATURE_NAMES:
            set_string = set_string + f + ' = ?,'
        set_string = set_string[0:-1]
        for i, segment in enumerate(segments_to_update):
            features = [segment.get_metadata('features_dict')[key] for key in FEATURE_NAMES]
            query = 'UPDATE tbl_segments SET %s WHERE ' \
                    'id_im = ? AND ' \
                    'coor_lu_x = ? AND ' \
                    'coor_lu_y = ? AND ' \
                    'coor_rl_x = ? AND ' \
                    'coor_rl_y = ?' % set_string
            corners = segment.get_metadata('corners')
            id_im = segment.get_metadata('parent_im_id')
            features.extend([id_im, corners[0,0], corners[0,1],
                                            corners[2,0], corners[2,1]])
            self.cursor.execute(query, features)
        self.db.commit()

    def set_labels_desired(self, ids, labels):
        '''
        Set desired labels of segments.
        :param ids: ids of segments to set labels
        :param labels: list of dictitionaries or integers, for each id one entry. Integer values denotes classes as
        defined in the NUMBERS_LABELS_MAP variable. The use of the integer encoding is recommended.
        :return:
        '''
        for i,id_seg in enumerate(ids):
            # Extract keys and labels
            if type(labels[i]) is dict:
                lbl_names = labels[i].keys()
                lbl_values = [labels[i][key] for key in lbl_names]
                lbl_name = labels[i]['rough_shape']
                if labels[i]['shape_character'] is not 'none':
                    lbl_name = lbl_name+'_'+labels[i]['shape_character']
                label_number = LABELS_NUMBERS_MAP[lbl_name]
            if type(labels[i]) is int:
                label_number = labels[i]
            query = "SELECT count(*) FROM tbl_segments WHERE id_seg=?"
            self.cursor.execute(query, (id_seg,))
            data = self.cursor.fetchone()[0]
            if data > 0:
                print("Updating label...")
                query = "UPDATE tbl_segments SET label_shape_desired=? WHERE id_seg=?"
                self.cursor.execute(query, (label_number, id_seg,))

            if data == 0:
                print('DataManager.set_label_desired(): segment not in database. Cannot label it.')
        self.db.commit()

    def get_labels_desired(self, ids):
        """
        Extract labels from the database.
        :param ids: list of ids if segments
        :return: annotated labels (integers) of given segments. See NUMBERS_LABELS_MAP for the meaning of numbers.
        """

        out = []
        for id in ids:
            query = 'SELECT label_shape_desired FROM tbl_segments WHERE id_seg=?'
            self.cursor.execute(query, (id,))
            result = self.cursor.fetchall()
            if len(result) == 0:
                out.append(-1)
            else:
                out.append(result[0][0])

        return np.array(out)

    def set_labels_classified(self, ids, labels, prob=None):
        '''
        Set desired (annotated) labels of segments.
        :param ids: ids of segments to set labels, list or numpy array [n_samples]
        :param labels: list of dictitionaries or integers. Dictitionaries are supposed to be used only for compatibility
        reasons with older code. The use of integers is strongly recommended.
        :param prob: numpy array of class probabilites [n_samples, n_classes]. Default is None.
        :return:
        '''
        for i,id_seg in enumerate(ids):
            # Extract keys and labels
            if type(labels) is dict:
                lbl_names = labels[i].keys()
                lbl_values = [labels[i][key] for key in lbl_names]
                lbl_name = labels[i]['rough_shape']
                if labels[i]['shape_character'] is not 'none':
                    lbl_name = lbl_name+'_'+labels[i]['shape_character']

                label_number = LABELS_NUMBERS_MAP[lbl_name]
            if type(labels) is list or type(labels) is tuple or type(labels) is np.ndarray:
                label_number = labels[i]

            query = "SELECT count(*) FROM tbl_segments WHERE id_seg=?"
            self.cursor.execute(query, (id_seg,))
            data = self.cursor.fetchone()[0]
            if data > 0:
                if prob is None:
                    prob = -1 * np.ones((1, len(POSSIBLE_LABELS)))
                #print("Updating label...")
                set_names = 'label_shape_classified=?,'
                query_values = [label_number]
                for k,l in enumerate(POSSIBLE_LABELS[1:]):
                    set_names = set_names+'p_cl_'+str(k+1)+'=?,'
                    query_values.append(prob[i,k])
                set_names=set_names[0:-1]
                query_values.append(id_seg)

                query = "UPDATE tbl_segments SET " + set_names + "WHERE id_seg=?"
                self.cursor.execute(query, query_values)


            if data == 0:
                print('DataManager.set_labels_classified(): segment not in database. Cannot label it.')
        self.db.commit()

    def get_labels_classified(self, ids):
        '''
        :param ids: List of ids of segments
        :return: List of integer labels of each segment. See NUMBERS_LABELS_MAP for the meaning of integer labels.
        '''
        out = []
        for id in ids:
            query = 'SELECT label_shape_classified FROM tbl_segments WHERE id_seg=?'
            self.cursor.execute(query, (id,))
            result = self.cursor.fetchall()
            if len(result) == 0:
                out.append(-1)
            else:
                out.append(result[0][0])
        return np.array(out)

    def get_class_counts(self, im_ids):
        '''
        :param im_ids: ids of full resolution images
        :return: matrix n_images x n_classes with counts of segments from different classes for each queried image
        '''
        classes = [i+1 for i,_ in enumerate(POSSIBLE_LABELS[1:])]
        counts = np.array(np.zeros((len(im_ids),len(classes))))
        for i,id in enumerate(im_ids):
            print("getting classes for im_id: {}".format(id))
            for j,cl in enumerate(classes):
                query = 'SELECT COUNT(*) FROM tbl_segments WHERE id_im = ? ' \
                        'AND label_shape_classified = ?'
                self.cursor.execute(query, (id, cl))
                counts[i, j] = self.cursor.fetchone()[0]
        return counts
    
    def get_geo_data(self, im_ids):
        '''
        :param im_ids: ids of images
        :return: numpy matrix (n_images x 4), the columns contains latitude, longitude (both in degrees), time when the
        image was created (in the Unix format) and length of exposure.
        '''
        lat_long_time_exp = np.zeros((len(im_ids), 4))
        for i,id in enumerate(im_ids):
            query = 'SELECT latitude, longitude, time, exposure FROM tbl_images WHERE id_im = ? '
            self.cursor.execute(query, (id,))
            lat, lon, time, exp = self.cursor.fetchone()
            lat_long_time_exp[i, :] = np.array([lat, lon, time, exp])
        return lat_long_time_exp

    def find_impaths(self, im_folder, file_type='vzlusat;pkl;01'):
        """
        Looks into im_folder and get all loadable files
        :im_folder: folder where to look for files
        :file_type: string describing frames to load
        currently supported:
        'vzlusat;pkl;XY', where XY is postprocessing mode: '01','02','04','08','16' or '32'
        'vzlusat;pkl;all': look for all frames, not only full resolution...
        :return: list of filenames, which can be passed to function load_images as argument im_paths
        """
        im_names = listdir(im_folder)
        im_names_filtered = list()
        file_type = file_type.split(';')
        for name in im_names:
            ending = name.split('.')[-1]
            name_without_ending = name.split('.')[0]
            if file_type[0] == 'vzlusat':
                if ending == 'pkl' :
                    vzlusat_type = name_without_ending.split('_')[-1]
                    if vzlusat_type == file_type[-1]:
                        im_names_filtered.append(name)
                    elif file_type[-1] == 'all':
                        im_names_filtered.append(name)

        return im_names_filtered

    def generate_placeholders(self, nr_placeholders):
        '''
        Utility function for internal use in the DataManager
        :param nr_placeholders:
        :return:
        '''
        l = range(nr_placeholders)
        placeholder = '?'  # For SQLite. See DBAPI paramstyle.
        placeholders = ', '.join(placeholder for unused in l)
        return placeholders

    def generate_seg_props(self, fnames=FEATURE_NAMES):
        '''
        Utility function for internal use in the DataManager
        '''
        fnames = [name + ',' for name in fnames]
        fnames =''.join(fnames)[0:-1]
        cols = ["id_im, coor_lu_x, coor_lu_y, coor_rl_x, coor_rl_y, "
                "label_shape_desired, label_shape_classified, " + fnames]
        cols = ''.join(cols)
        return cols

    def nolearn_tag(self, id, comments_string):
        '''
        Utility function for internal use in the DataManager
        '''
        spl = comments_string.split('\n')
        ids = spl[0::2]
        comments = spl[1::2]
        index = ids.index(str(id))
        if comments[index].find('#nolearn') > -1:
            print("problem: im.id: {}".format(id))
            return 1
        else:
            return 0


def make_corners_string(corners):
    '''
    Utility function for internal use in the DataManager
    '''
    corners_str = str(corners[0, 0]) + ', ' + str(corners[0, 1]) + ';' + \
                  str(corners[1, 0]) + ', ' + str(corners[1, 1]) + ';' + \
                  str(corners[2, 0]) + ', ' + str(corners[2, 1]) + ';' + \
                  str(corners[3, 0]) + ', ' + str(corners[3, 1]) + ';'
    return corners_str

