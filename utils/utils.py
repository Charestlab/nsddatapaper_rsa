import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform


def mds(utv, pos=None, n_jobs=1):
    """ pos = mds(utv)

    mds computes the multi-dimensional scaling solution on a 
    two dimensional plane, for a representational dissimilarity matrix.

    Args:

        utv (array): 1D upper triangular part of an RDM

        pos (array, optional): set of 2D coordinates to initialise the MDS
                            with. Defaults to None.

        n_jobs (int, optional): number of cores to distribute to.
                            Defaults to 1.

    Returns:

        [array]: 2D aray of x and y coordinates.

    """

    rdm = squareform(utv)
    seed = np.random.RandomState(seed=3)
    mds = MDS(
        n_components=2,
        max_iter=100,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=n_jobs
    )
    pos = mds.fit_transform(rdm, init=pos)

    return pos


# this filters images with animate items.
category_dict = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 1,
    '7': 0,
    '8': 0,
    '9': 0,
    '10': 1,
    '11': 0,
    '12': 0,
    '13': 0,
    '14': 0,
    '15': 0,
    '16': 0,
    '17': 0,
    '18': 0,
    '19': 0,
    '20': 1,
    '21': 0,
    '22': 0,
    '23': 0,
    '24': 0,
    '25': 1,
    '26': 0,
    '27': 0,
    '28': 1,
    '29': 0,
    '30': 1,
    '31': 0,
    '32': 0,
    '33': 0,
    '34': 1,
    '35': 0,
    '36': 0,
    '37': 1,
    '38': 0,
    '39': 0,
    '40': 0,
    '41': 0,
    '42': 0,
    '43': 0,
    '44': 0,
    '45': 1,
    '46': 0,
    '47': 0,
    '48': 0,
    '49': 1,
    '50': 0,
    '51': 0,
    '52': 0,
    '53': 0,
    '54': 0,
    '55': 0,
    '56': 1,
    '57': 0,
    '58': 0,
    '59': 0,
    '60': 1,
    '61': 0,
    '62': 0,
    '63': 0,
    '64': 0,
    '65': 0,
    '66': 0,
    '67': 0,
    '68': 0,
    '69': 0,
    '70': 0,
    '71': 0,
    '72': 0,
    '73': 0,
    '74': 0,
    '75': 0,
    '76': 0,
    '77': 0,
    '78': 0,
    '79': 1
 }


def average_over_conditions(data, conditions, conditions_to_avg):
    
    lookup =  np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims==2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape 
        avg_data = np.empty((x,y,z, n_conds))

    for j, x in enumerate(lookup):

        conditions_bool = conditions==x 
        if n_dims ==2:
            avg_data[:,j] = np.nanmean(data[:, conditions_bool], axis=1)
        else:
            avg_data[:,:,:,j] = np.nanmean(data[:, :, :, conditions_bool], axis=3)

    return avg_data
