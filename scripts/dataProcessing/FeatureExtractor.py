'''
Class which performs feature extraction.

It is possible to modify the set of calculated features by modification of
the EXTRACTORS_LIST variable. This is a tuple which contains triplets. Each
triplet has a following structure: ("name_of_the_feature", feature extraction function,
keyword arguments). For description of the feature extraction function please refer to
the file extractionFcns.py.
'''

from extractionFcns import *

EXTRACTORS_LIST = (("area", area, {}),
                   ("energy", energy, {}),
                   ("nr_crossings", nr_crossings, {}),
                   ("chull_area", chull_area, {}),
                   ("pixel_energy", pixel_energy, {}),
                   ("chull_occupancy", chull_occupancy, {}),
                   ("linear_fit", linear_fit, {}),
                   ("skeleton_chull_ratio", skeleton_chull_ratio , {}),
                   ("pixels_direction", pixels_direction ,{}),
                   ("tortuosity", tortuosity, {}),
                   ("dist_mean", dist_mean, {}),
                   ("dist_std", dist_std, {}),
                   ("eig_ratio", eig_ratio, {}),
                   ("boxiness", boxiness, {}),
                   ("diagonaliness", diagonaliness, {}),
                   ("straightness", straightness, {}),
                   ("q10", q10, {}),
                   ("q50", q50, {}),
                   ("q90", q90, {}),
                   ("e1", e1, {}),
                   ("e2", e2, {}),
                   ("e3", e3, {}),
                   ("e4", e4, {}),
                   ("bheigth", bheigth, {}),
                   ("bwidth", bwidth, {}))

class FeatureExtractor():
    def extract_features(self, segments_collection, extractors_list=EXTRACTORS_LIST):
        """
        Perform feature extraction for a list of segments. The list is modified in-place to
        conserve memory.
        :param segments_collection: list of segments for which the features should be calculated
        :param extractors_list: ((name1, fcn1, dict_of_arguments1),
        (name2, fcn2, dict_of_arguments2),...)
        fcn must look like: fcn(bitmap[mxn], **kwargs) and must return feature value
        :return:
        """
        for i,image in enumerate(segments_collection):
            feat_dict = dict()
            for extractor_tuple in extractors_list:
                name = extractor_tuple[0]
                fcn = extractor_tuple[1]
                kwargs = extractor_tuple[2]
                feature_value = fcn(image.get_bitmap(), **kwargs)
                feat_dict[name] = feature_value
            image.set_metadata(key='features_dict', data=feat_dict)
