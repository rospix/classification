'''
Various functions used during feature exploration.
'''
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
from dataManager.DataManager import FEATURE_NAMES
from dataManager.DataManager import POSSIBLE_LABELS
    
def feat_distribution(X, y, feat_vs_feat, classes=(4,), n_ax = (2,1),feat_names=FEATURE_NAMES, class_names=POSSIBLE_LABELS[1:]):
    fig, axarr = plt.subplots(n_ax[0], n_ax[1])
    axarr = axarr.flatten()
    
    for i in range(0, len(feat_vs_feat)):
        for cl in classes:
            jitterx = np.random.uniform(-0.3,0.3,np.count_nonzero(y==cl))
            jittery = np.random.uniform(-0.3,0.3,np.count_nonzero(y==cl))
            axarr[i].scatter(X[y==cl,feat_vs_feat[i][0]]+jitterx, X[y==cl, feat_vs_feat[i][1]]+jittery, 
                             alpha=0.3,s=3)
            axarr[i].set_xlabel(FEATURE_NAMES[feat_vs_feat[i][0]])
            axarr[i].set_ylabel(FEATURE_NAMES[feat_vs_feat[i][1]])
            #axarr[i].set_ylabel(FEATURE_NAMES[feat_vs_feat[i][1]])
        axarr[i].legend([POSSIBLE_LABELS[cl] for cl in classes])
        #axarr[i].scatter(X[:,feat_vs_feat[0]], X[:, feat_vs_feat[1]])
   
def scatterplot_matrix(X, y):
    fig, axes = plt.subplots(X.shape[1], X.shape[1])
    plots = list()
    #colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(0, X.shape[1]):
        for j in range(0, X.shape[1]):
            for cl in range(0, np.max(y)):
                if i==0 and j==0:
                    label=POSSIBLE_LABELS[cl+1]   
                    plots.append(axes[i,j].scatter(X[y==cl+1,i], X[y==cl+1,j], s=1, alpha=0.5, label=label))
                plots.append(axes[i,j].scatter(X[y==cl+1,i], X[y==cl+1,j], s=1, alpha=0.5))
                axes[i,j].set_xticks([],[])
                axes[i,j].set_yticks([],[])
                
    #h, l = axes[2,2].get_legend_handles_labels()
    #axes[2,2].legend(h,l, borderaxespad=0)
    #axes[2,2].axis("off")    
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.legend()
    plt.show()
    
def hist(X, y, cl, feature, bins):
    # Create histogram for specified class and feature
    # (= P(cl joint feature))
    class_indices = np.nonzero(y[:,cl]==1)
    histogram, bin_edges = np.histogram(X[class_indices, feature], bins=10)
    return histogram, bin_edges

def get_features_of_class(X, y, cl):
    class_indices = np.flatnonzero(y[:, cl] == 1)
    return X[class_indices, :]

def feature_histograms_regression(X, y, plot=True):
    fig, axes = plt.subplots(y.shape[1],1, sharex=True, sharey=True)
    hists = list()
    bs = list()
    for cl in range(0, y.shape[1]):
        for feat in range(0, X.shape[1]):
            xf = get_features_of_class(X, y, cl)[:,feat]
            axes[cl].scatter(xf,np.random.uniform(feat-0.2, feat+0.2, size=(xf.shape[0],1)),
                             s=0.5)
    plt.tight_layout()
    if plot==True:
        plt.show()
        
def feature_histograms_classif(X, y, plot=True):
    fig, axes = plt.subplots(int(np.ceil(X.shape[1]/2))+1,2, sharey=True)
    axes=axes.flatten()
    hists = list()
    bs = list()
    for cl in range(1, np.max(y)+1):
        for feat in range(0, X.shape[1]):
            xf = X[y==cl,feat]
            axes[feat].scatter(xf, np.random.uniform(-0.3,0.3,xf.shape)+cl*np.ones((1,xf.shape[0])), 
                               s=2, alpha=0.3)
            #h, bins = np.histogram(xf)
            #centers = [(bins[i]+bins[i+1])/2 for i in range(0, bins.shape[0]-1)]
            #axes[cl-1].bar(h, centers)
    plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)
    if plot==True:
        plt.show()

def feat_histogram_simple(X, plot=True):
    fig, axes = plt.subplots(int(np.ceil(X.shape[1]/2)),2)
    axes = axes.flatten()
    hists = list()
    bs = list()

    for feat in range(0, X.shape[1]):
        axes[feat].hist(X[:,feat], bins=100, histtype='stepfilled')
        #axes[feat].hist(preprocessing.minmax_scale(X[:,feat].reshape((-1,1)), output_distribution='normal'), 
        #                bins=100, histtype='step')
        axes[feat].set_yscale('log', nonposy='clip')
        #kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit((X[:,feat])[:, np.newaxis])
        #X_plot = np.linspace(np.min(X[:,feat]), np.max(X[:,feat]), 100)[:, np.newaxis]
        #log_dens = kde.score_samples(X_plot)
        #h, bins = np.histogram(X[:,feat], bins='auto')
        #centers = [(bins[i]+bins[i+1])/2 for i in range(0, bins.shape[0]-1)]
        #axes[feat].bar(centers, h)
        #axes[feat].fill_between(X_plot[:, 0], np.exp(log_dens), np.zeros((X_plot.shape[0],1)).flatten())
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    if plot==True:
        #plt.gca().invert_yaxis()
        plt.show()    

def variance_plot(X, plot=True):
    plot_feature_scores(feature_names=FEATURE_NAMES,
                        feature_scores=np.var(X,axis=0),
                        y_name='Variance [-]',
                        plot_name='Variance plot',
                        plot=plot)


def mutual_info_plot(X, y, plot=True):
    mi = np.zeros((int(np.max(y)), X.shape[1]))
    fig, axes = plt.subplots(y.shape[1], 1)
    for cl in range(0, y.shape[1]):
        mi[cl] = mutual_info_classif(X, y[:, cl])
        if plot is True:
            plot_feature_scores(feature_names=FEATURE_NAMES,
                                feature_scores=mi[cl],
                                y_name='Mutual information')
            axes[cl].stem(mi[cl])
            #axes[cl].set_xlabel(FEATURE_NAMES[j], fontsize=8)
            axes[cl].set_aspect('auto')
            #axes[cl].tick_params(labelsize=8)
            #plt.tight_layout(pad=1.08, h_pad=0, w_pad=0, rect=[0, 0, 1, 1])
    if plot is True:
        plt.legend()
        plt.show()
    return mi


def features_1d_plots(X, y, plot=True):
    for cl in range(y.shape[1]):
        fig, axes = plt.subplots(1, int(np.ceil(X.shape[1])))
        print9cl)
        fig.suptitle(POSSIBLE_LABELS[cl+1])
        axf = axes.flatten()        
        for i in range(0, X.shape[1]):
            colors = np.ones((300,3))
            colors[:,0] = y[:,cl];colors[:,1] = y[:,cl];colors[:,2] = y[:,cl]
            colors = (colors-np.min(colors))/(np.max(colors-np.min(colors)))
            axf[i].scatter(np.random.uniform(0, 1, X.shape[0]), 
                           X[:,i], c=colors, s=5, edgecolors='none')
            axf[i].set_xticks([])
            axf[i].set_yticks([])
            axf[i].autoscale_view()
    if plot==True:
        plt.show()
        

def model_based_selection(X,y,selector=ExtraTreesClassifier(), plot=True, feature_names=FEATURE_NAMES):
    selector.fit(X, y)
    print(selector.feature_importances_)
    plot_feature_scores(feature_names,
                        selector.feature_importances_,
                        'Feature scores', plot_name='Tree-computed feature scores', plot=plot)

def feat_class_matrix(X, y, fcn, n_classes, plot=True):
    mat = np.zeros((X.shape[1], n_classes))
    for feat in range(X.shape[1]):
        for cl in range(0, n_classes):
            print9np.corrcoef((X[y==cl+1,feat], y[y==cl+1])))
            mat[feat, cl] = fcn(X[y==cl+1,feat], y[y==cl+1])
   
    plt.imshow(mat)     
    if plot==True:
        plt.show()

def plot_feature_scores(feature_names, feature_scores, y_name, plot_name='Feature scores', plot=True):
    plt.figure(plot_name)
    importances = feature_scores
    idxs = np.argsort(importances)
    importances = importances[idxs]
    plt.barh(range(0, feature_scores.shape[0]), importances)

    names = [fname.replace('_', '\_') for fname in feature_names]
    names = [names[id] for id in idxs]
    print9names)

    plt.yticks(range(0, feature_scores.shape[0]), names)
    plt.xlabel('Feature importance [-]')
    if plot==True:
        plt.show()
