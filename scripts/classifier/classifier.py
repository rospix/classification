'''
Various classification experiments and ideas used during prototyping (not used in the final solution):
-> Classification of segments with the use of bank of filters
-> Semisupervised learning
-> Class-based classifier interface
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from plotting.learningcurves import plot_validation_curve, plot_learning_curve
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, jaccard_similarity_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, MaxPool2D, AvgPool2D
from keras.callbacks import EarlyStopping

from skimage.morphology import erosion
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

from plotting.error_eval import crossval_multiclass, confusion_matrix_multiclass

from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.model_selection import learning_curve
from sklearn import model_selection
from sklearn import metrics

import random

class Filter_dot(BaseEstimator, ClassifierMixin):
    def fit(self, X=None, y=None):
        return self
    def predict(self, X):
        y = np.zeros((X.shape[0], 1))
        y[X[:,0]<=2] = 1
        return y


class Classifier_complex_old(BaseEstimator, ClassifierMixin):
    def __init__(self, est=OneVsRestClassifier(RandomForestClassifier())):
        self.model = OneVsRestClassifier(RandomForestClassifier())
    def fit(self, X, y):
        X = X[y!=1]
        y = y[y!=1]
        self.model.fit(X, y)
        return self
    def predict(self, X, y=None):
        y = np.zeros((X.shape[0], 1))
        dots_mask = X[:,0]<3
        p = self.model.predict(X[np.logical_not(dots_mask),:])
        print9p,(p.shape0
        y[np.logical_not(dots_mask)] = p
        y[dots_mask] = 1
        return np.zeros((X.shape[0], 1))


class Classifier_semisupervised(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.X_labeled = ''
        self.y_labeled = ''

    def fit(self, X, y):
        self.X_labeled = X
        self.y_labeled = np.reshape(y, (-1,1))
        return self

    def predict(self, X):
        ls = LabelPropagation()
        X_unknown = X
        y = np.ravel(np.vstack((self.y_labeled, -np.ones((X_unknown.shape[0], 1)))))
        X = np.vstack((self.X_labeled, X_unknown))
        ls.fit(X, y)
        y_p = ls.predict(X_unknown)
        print9y_p)
        return y_p

class Classifier_complex(BaseEstimator, ClassifierMixin):
    def __init__(self, est=RandomForestClassifier()):
        self.est = est
        self.models = list()
        self.s = StandardScaler()
        pf = PolynomialFeatures(degree=3)
        self.one_class_pipeline = Pipeline(steps=[('pf',pf),
                                                  ('est', self.est)])
        self.big_pipeline = OneVsRestClassifier(estimator=self.one_class_pipeline)

    def fit(self, X, y):
        n_classes = np.max(y).astype(dtype=np.int)
        self.s = RobustScaler()
        self.s.fit(X, y)
        X = self.s.transform(X)
        self.big_pipeline.fit(X, y)
        return self

    def predict(self, X):
        X = self.s.transform(X)
        y = np.zeros((X.shape[0], 1))
        y = self.big_pipeline.predict(X)
        return y

    def learning_curve(self, X, y, sizes=np.linspace(0.1,1,5),n=2):
        tr_scores = np.zeros((sizes.shape[0], np.max(y), n))
        v_scores = np.zeros((sizes.shape[0], np.max(y), n))
        for i,relsize in enumerate(sizes):
            size = np.round(relsize*X.shape[0]).astype(dtype=int)
            print9size)
            ind = np.array(range(0, X.shape[0]))
            random.shuffle(ind)
            print9ind)
            ind = ind[range(0, size)]
            val_scores, train_scores = self.crossvalidation(X[ind, :], y[ind], n)
            v_scores[i,:,:] = val_scores
            tr_scores[i,:,:] = train_scores

        v_scores = np.mean(v_scores, axis=2)
        tr_scores = np.mean(tr_scores, axis=2)

        fig, ax = plt.subplots(5,2)
        ax = ax.flatten()
        for i,model in enumerate(self.big_pipeline.estimators_):
            l1, = ax[i].plot(sizes*X.shape[0], tr_scores[:,i], label='Training')
            l2, = ax[i].plot(sizes*X.shape[0], v_scores[:,i], label='Testing')
            ax[i].legend(handles=[l1, l2])
            ax[i].set_ylabel('MCC')
        plt.show()

    def crossvalidation(self, X, y, n):
        cv = model_selection.StratifiedKFold(n_splits=n)
        cv.get_n_splits(X,y)
        scores = np.zeros((np.max(y), n)) # (n_classes, ncv)
        train_scores = np.zeros((np.max(y), n)) # (n_classes, ncv)
        i = 0
        for train_index, test_index in cv.split(X, y):
            print9train_index.shape,(X.shape0
            X_train = X[train_index,:]
            y_train=y[train_index]
            X_test = X[test_index,:]
            y_test=y[test_index]
            self.big_pipeline.fit(X_train, y_train)
            y_train_pred = self.big_pipeline.predict(X_train)
            y_pred = self.big_pipeline.predict(X_test)

            scores[:,i] = np.array([metrics.matthews_corrcoef(y_test==j, y_pred==j) for j in range(0, np.max(y))])
            train_scores[:,i] = np.array([metrics.matthews_corrcoef(y_train==j, y_train_pred==j) for j in range(0, np.max(y))])
            i = i+1
        return scores, train_scores

def build_sklearn_keras():
    model = Sequential()
    model.add(Reshape((32,32,1), input_shape=(32,32)))
    model.add(Conv2D(filters=5, kernel_size=2, padding='same'))
    model.add(Dense(2, activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=2, padding='same'))
    model.add(AvgPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=5, kernel_size=2, padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.layers[1].trainable=True
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

class Classifier_filters(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.call = EarlyStopping(monitor='acc_loss', min_delta=0.01, patience=2, verbose=1)
        self.clf = KerasClassifier(build_fn=build_sklearn_keras,
                                       epochs=100, batch_size=50, verbose=1)
        self.clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=200))

    def fit(self, X, y):
        X2 = np.zeros((X.shape[0], X.shape[1]*X.shape[2]))
        for i in range(0,X.shape[0]):
            X2[i,:] = X[i,:,:].flatten()
        self.clf.fit(X2, y)
        return self

    def predict(self, X, y=None):
        X2 = np.zeros((X.shape[0], X.shape[1]*X.shape[2]))
        for i in range(0,X.shape[0]):
            X2[i,:] = X[i,:,:].flatten()
        return self.clf.predict(X2)


class FilterBank():
    def __init__(self):
        self.filters=list()
        for i in range(0,20):
            self.filters.append(np.reshape(np.random.randint(0,2,9), (3,3)))
            print(self.filters[-1])

    def predict(self, X):
        X2 = np.zeros((X.shape[0],X.shape[1],X.shape[2],len(self.filters)))
        X3 = np.zeros((X.shape[0], len(self.filters)))
        for im in range(0, X.shape[0]):
            for i,f in enumerate(self.filters):
                X2[im,:,:,i] = erosion(X[im, :, :], selem=self.filters[i])
                X3[im, i] = np.sum(X2[im,:,:,i])
        return X3

