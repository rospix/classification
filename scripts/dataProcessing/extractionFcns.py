'''
This file contains various feature extractors which are used by the FeatureExtractor class.
All the functions must have the same interface: first input attribute is a 2D numpy array
containing image, other arguments, if needed, can be added as keyword arguments. Each
calculated feature can be only 1D and its datatype must be supported by the database backend
of the DataManager (currently SQLite).
'''

import numpy as np
from skimage.morphology import skeletonize, convex_hull_image, binary_erosion, binary_opening, binary_closing, binary_dilation
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_cdt
import networkx as nx

def area(image):
    '''
    Count nonzero pixels
    :param image: 2D numpy array
    :return: count of nonzeros pixels
    '''
    return int(np.count_nonzero(image))

def energy(image):
    '''
    :param image:
    :return:
    '''
    return int(np.sum(image))

def pixel_energy(image):
    return np.sum(image)/np.count_nonzero(image)

def chull_occupancy(image):
    return float(np.count_nonzero(image))/float(chull_area(image))

def nr_crossings(image, method="graph"):
    '''
    Count number of branches
    :param image:
    :return: number of branches in image
    '''
    im = np.copy(image)
    im[im > 0] = 1
    im_skeleton = skeletonize(np.atleast_2d(im))

    if method == "graph":
        if im_skeleton.shape[0] > 2 and im_skeleton.shape[1] > 2:
            G = bin2graph(im_skeleton)
            G = nx.minimum_spanning_tree(G)

            # G = nx.dodecahedral_graph()

            crossings = list()
            for node in G:
                crossings.append(len(G.edges(node)))
            crossings = np.array(crossings)
            crossings = len(crossings[crossings > 2])
            #if crossings > 0:
            #    p = nx.spring_layout(G)
            #    nx.draw(G, p, with_labels=True)
            #    nx.draw_networkx_edge_labels(G, pos=p)
            #    plt.show()
            #    print im_skeleton
        else:
            crossings = 0
    else:
        rows = im_skeleton.shape[0]
        cols = im_skeleton.shape[1]
        cr_list = []
        if rows < 3 and cols < 3:
            return 0
        crossings = 0
        for row in range(1,rows-1):
            for col in range(1,cols-1):
                if im_skeleton[row, col] == True:
                    if np.sum(im_skeleton[row-1:row+2, col-1:col+2]) > 3:
                        crossings = crossings+1
                        cr_list.append((row, col))
    #if crossings > 0:
    #    print "******"
    #    print im_skeleton
    #    print "Crossings detected: ", crossings
    return crossings

def fractal_dimension(image):
    im = np.copy(image)
    # increase resolution



def skeleton(image):
    #im = np.copy(image)
    #im[im > 0] = 1
    im_skeleton = skeletonize(image>0)
    return im_skeleton

def chull_area(image):
    #print(image)
    #im = np.copy(image)
    #print im
    #im[im > 0] = 1
    hull = convex_hull_image(image>0)
    return np.count_nonzero(hull)

def linear_fit(image):
    '''
    This function finds general line equation and reutrn coefficient of determination
    http://cmp.felk.cvut.cz/cmp/courses/XE33PVR/WS20072008/Lectures/Supporting/constrained_lsq.pdf
    :param image:
    :return:
    '''
    if image.shape[0] < 2 and image.shape[1] < 2:
        return 0.0


    rr, cc = np.nonzero(image)
    centerR = np.mean(rr)
    centerC = np.mean(cc)
    center = np.array([[centerR],[centerC]])
    rr = rr-center
    cc = cc-center
    S = np.cov(np.vstack((rr, cc)))
    nrs, vecs = np.linalg.eig(S)
    nrs = np.absolute(nrs)
    max_eig = np.max(nrs)
    explained_variance_ratio = max_eig/np.sum(nrs)

    return explained_variance_ratio

def skeleton_chull_ratio(image):
    sk = skeleton(image)
    return np.sum(sk)/chull_area(image)


def pixels_direction(image):
    if image.shape[0] == 1 and image.shape[1] == 1:
        return 0.0
    sk = skeleton(image)
    left_right = 0
    top_bottom = 0
    diag_rd = 0
    diag_ru = 0
    for row in range(0, image.shape[0]):
        for col in range(0, image.shape[1]-1):
            left_right = left_right + sk[row, col] * sk[row, col+1]
    for row in range(0, image.shape[0]-1):
        for col in range(0, image.shape[1]):
            top_bottom = top_bottom + sk[row, col] * sk[row+1, col]
    for row in range(0, image.shape[0]-1):
        for col in range(0, image.shape[1]-1):
            diag_rd = diag_rd + sk[row, col] * sk[row+1, col+1]
    for row in range(1, image.shape[0]):
        for col in range(0, image.shape[1]-1):
            diag_ru = diag_ru + sk[row, col] * sk[row-1, col+1]

    a = area(sk)
    fraction_left_right = float(left_right)/float(a)
    fraction_top_bottom = float(top_bottom)/float(a)
    fraction_diag_rd = float(diag_rd) / float(a)
    fraction_diag_ru = float(diag_ru) / float(a)
    return np.max([fraction_left_right, fraction_top_bottom, fraction_diag_rd, fraction_diag_ru])

def tortuosity(image):
    if image.shape[0] == 1 and image.shape[1] == 1:
        return 0.0
    sk = skeleton(image)
    gr = bin2graph(sk, diagonal_penalty=2)
    spt = nx.minimum_spanning_tree(gr)
    start = list(spt.nodes())[0]
    if len(spt.nodes()) <= 1:
        return 0.0
    #print start
    #print image
    #print(sk)
    #print(list(nx.bfs_edges(spt, start)))
    u = list(nx.bfs_edges(spt, start))[-1][-1]
    v = list(nx.bfs_edges(spt, u))[-1][-1]
    longest_path = nx.shortest_path_length(spt, u, v, weight='weight')
    start = np.array(u)
    end = np.array(v)
    start_end_length = np.linalg.norm(start-end)
    return float(longest_path)/start_end_length

def dist_mean(im):
    dist = distance_transform_cdt(im, metric='chessboard', return_indices=False)
    #axis, dist = medial_axis(im, return_distance=True)
    return np.mean(dist)

def dist_std(im):
    dist = distance_transform_cdt(im, metric='chessboard', return_indices=False)
    #axis, dist = medial_axis(im, return_distance=True)
    return np.std(dist)

def eig_ratio(image):
    if image.shape[0] == 1 and image.shape[1] == 1:
        return 1.0
    rr, cc = image.nonzero()
    m = np.vstack((rr, cc))
    S = np.cov(m=m)
    vals,_ = np.linalg.eig(S)
    return np.min(vals)/np.max(vals)


def boxiness(image):
    bim = image > 0   
    bim0 = np.zeros(bim.shape)
    bim1 = np.zeros(bim.shape)
    bim1 = bim
    selem = np.array([[1,1,0],
                      [1,1,0],
                      [0,0,0]])
    # one erosion to remove everything which is smaller than square
    bim1 = binary_erosion(bim1, selem)
    # thinning
    bim1 = skeletonize(bim1)
    # count connected components
    li, nlbls = label(bim1, return_num=True)
    return nlbls

def diagonaliness(image):
    bim = image > 0
    selem = np.array([[1,1,0],
                      [1,1,0],
                      [0,0,0]])

    small_deleted = binary_opening(bim,selem)
    large_deleted = np.bitwise_xor(bim, small_deleted)
    diagonal_detected = np.bitwise_or(binary_opening(large_deleted, np.array([[1,0,0],
                                                                [0,1,0],
                                                                [0,0,0]])),
                                      binary_opening(large_deleted, np.array([[0,0,1],
                                                                              [0,1,0],
                                                                              [0,0,0]])))    

    return np.sum(diagonal_detected)

def straightness(image):

    bim = image > 0

    selem = np.array([[1,1,0],
                      [1,1,0],
                      [0,0,0]])

    small_deleted = binary_opening(bim, selem)
    large_deleted = np.bitwise_xor(bim, small_deleted)
    straight_detected = np.bitwise_or(binary_opening(large_deleted, np.array([[1,1,0]])),
                                      binary_opening(large_deleted, np.array([[1],[1],[1]])))    

    return np.sum(straight_detected)

def q10(image):
    return np.percentile(image, 10)

def q50(image):
    return np.percentile(image, 50)

def q90(image):
    return np.percentile(image, 90)

def e1(image):
    bim = image>0
    e1 = np.zeros(image.shape).astype(dtype='bool')
    e2 = np.zeros(image.shape)
    e3 = np.zeros(image.shape)
    e4 = np.zeros(image.shape)
    # detect pixels which are connected with background only with one edge
    ses = [[[0,1,0],[1,1,0],[0,1,0]],
           [[0,1,0],[1,1,1],[0,0,0]],
           [[0,1,0],[0,1,1],[0,1,0]],
           [[0,0,0],[1,1,1],[0,1,0]]]
    ses2 = [[[0,0,0],[0,0,1],[0,0,0]],
           [[0,0,0],[0,0,0],[0,1,0]],
           [[0,0,0],[1,0,0],[0,0,0]],
           [[0,1,0],[0,0,0],[0,0,0]]]    
    for i,se in enumerate(ses):
        e1 = np.bitwise_or(e1, hmtransform(bim, se, ses2[i]))    
    # detect pixels connected with background with two edges
    ses = [[[1,1,1],
            [1,1,0],
            [1,0,0]],
           [[1,0,0],
            [1,1,0],
            [1,1,1]],
           [[0,0,1],
            [0,1,1],
            [1,1,1]],
           [[1,0,0],
            [1,1,0],
            [1,1,1]],
           [[0,1,0],
            [0,1,0],
            [0,1,0]],
           [[0,0,0],
            [1,1,1],
            [0,0,0]]]
    return np.sum(e1)


def e2(image):
    bim = image>0
    e2 = np.zeros(image.shape).astype(dtype='bool')  
    # detect pixels connected with background with two edges
    ses = [[[0,1,0],
            [1,1,0],
            [0,0,0]],
           [[0,0,0],
            [1,1,0],
            [0,1,0]],
           [[0,0,0],
            [0,1,1],
            [0,1,0]],
           [[0,0,0],
            [1,1,0],
            [0,1,0]],
           [[0,1,0],
            [0,1,0],
            [0,1,0]],
           [[0,0,0],
            [1,1,1],
            [0,0,0]]]
    ses2 = [[[0,0,0],
            [0,0,1],
            [0,1,0]],
           [[0,1,0],
            [0,0,1],
            [0,0,0]],
           [[0,1,0],
            [1,0,0],
            [0,0,0]],
           [[0,1,0],
            [0,0,1],
            [0,0,0]],
           [[0,0,0],
            [1,0,1],
            [0,0,0]],
           [[0,1,0],
            [0,0,0],
            [0,1,0]]]    
    for i,se in enumerate(ses):
        e2 = np.bitwise_or(e2, hmtransform(bim, se, ses2[i]))       
    return np.sum(e2)

def e3(image):
    bim = image>0
    e3 = np.zeros(image.shape).astype(dtype='bool')     
    # detect pixels connected with three edges
    ses = [[[0,1,0],
                [0,1,0],
                [0,0,0]],
               [[0,0,0],
                [1,1,0],
                [0,0,0]],
               [[0,0,0],
                [0,1,0],
                [0,1,0]],
               [[0,0,0],
                [0,1,1],
                [0,0,0]]]
    ses2 = [[[0,0,0],
                [1,0,1],
                [0,1,0]],
               [[0,1,0],
                [0,0,1],
                [0,1,0]],
               [[0,1,0],
                [1,0,1],
                [0,0,0]],
               [[0,1,0],
                [1,0,0],
                [0,1,0]]]    
    for i,se in enumerate(ses):
        e3 = np.bitwise_or(e3, hmtransform(bim, se, ses2[i]))           
    return np.sum(e3)

def e4(image):
    bim = image>0
    e4 = np.zeros(image.shape).astype(dtype='bool')     
    # pixels with four edges
    ses = [[[0,0,0],
            [0,1,0],
            [0,0,0]]]
    ses2 = [[[0,1,0],
            [1,0,1],
            [0,1,0]]]    
    for i,se in enumerate(ses):
        e4 = np.bitwise_or(e4, hmtransform(bim, se, ses2[i]))         
    return np.sum(e4)

def bheigth(image):
    return image.shape[0]

def bwidth(image):
    return image.shape[1]

def hmtransform(bim, se1, se2):
    im2 = np.zeros((bim.shape[0]+2, bim.shape[1]+2)).astype(dtype='bool')
    im2[1:bim.shape[0]+1, 1:bim.shape[1]+1]=bim
    im2 = np.bitwise_and(binary_erosion(im2, selem=np.array(se1).astype(dtype='bool')), 
                   binary_erosion(np.bitwise_not(im2), 
                                  selem=np.array(se2).astype(dtype='bool')))    
    return im2[1:-1, 1:-1]


def fractal_dimension(Z, threshold=0):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform Z into a binary array
    Z = (Z > threshold)
    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def bin2graph(image, rc_penalty=1, diagonal_penalty=2, edge_length=1):
    visited = set()
    nonzero = set()
    G = nx.Graph()
    rows = image.shape[0]
    cols = image.shape[1]

    # Get coordinates of all nonzero points
    for row in range(0, rows):
        for col in range(0, cols):
            #print "Testing ", (row, col), image[row, col]
            if image[row, col] == True:
                #print (row, col), " is True"
                nonzero.add((row, col))
                G.add_node((row, col))

    # Iterate over all coordinates. For each point in image, if there is a point which differs by
    # 1 in some coordinate, make edge to it and put it into visited
    for s in nonzero:
        row = s[0]
        col = s[1]
        neighbors = [((row-1, col-1),diagonal_penalty),
                     ((row-1, col),rc_penalty),
                     ((row-1, col+1),diagonal_penalty),
                     ((row,col-1),rc_penalty),
                     ((row, col+1),rc_penalty),
                     ((row+1,col-1),diagonal_penalty),
                     ((row+1, col),rc_penalty),
                     ((row+1, col+1),diagonal_penalty)]
        for neighbor in neighbors:
            #print 'Testing ',s, " with ", neighbor
            if neighbor[0] in nonzero:
                #print neighbor[0]
                G.add_edge(s, neighbor[0], weight=neighbor[1], length=edge_length)
                #print s, ' connected with ', neighbor
    G=G.to_undirected()
    return G

if __name__ == "__main__":
    im_blob = np.array([[1,0,0,1,1],
                        [1,0,1,1,1],
                        [1,1,0,0,0],
                        [1,1,0,0,0]])
    im_blob = np.array([[1,1,0,0,1],
                        [1,1,1,1,1],
                        [1,0,1,0,0],
                        [1,0,0,1,0]])   
    
    selem = np.array([[1,1,0],
                      [1,1,0],
                      [0,0,0]])
    small_deleted = binary_opening(im_blob,selem)
    large_deleted = np.bitwise_xor(im_blob, small_deleted)
    diagonal_detected = binary_opening(large_deleted, np.array([[1,0,0],
                                                                [0,1,0],
                                                                [0,0,0]]))
    print diagonal_detected
    
    print(boxiness(im_blob))
    print diagonaliness(im_blob)
    print straightness(im_blob)
    im_blob = np.array([[1,1,1],[0,1,0],[0,1,0],[0,0,1]])
    print e1(im_blob)
    print e2(im_blob)
    print e3(im_blob)
    print e4(im_blob)
