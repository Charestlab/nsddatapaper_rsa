import sys
import os
import time
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist
from nsd_access import NSDAccess
from nsd_get_data import get_conditions, get_betas
from utils import average_over_conditions

"""
    module to gather the region of interest rdms
"""
sub = int(sys.argv[1])
n_jobs = 38
n_sessions = 40
n_subjects = 8

# set up directories
base_dir = os.path.join('/rds', 'projects', 'c')
nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
proj_dir = os.path.join(base_dir, 'charesti-start', 'projects', 'NSD')
nsd_dir = os.path.join(base_dir, 'charesti-start', 'data', 'NSD')
sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
betas_dir = os.path.join(proj_dir, 'rsa')
models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

# initiate nsd access
nsda = NSDAccess(nsd_dir)

# path where we save the rdms
outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

# we use the fsaverage space.
targetspace = 'fsaverage'

lh_file = os.path.join(proj_dir, 'lh.highlevelvisual.mgz')
rh_file = os.path.join(proj_dir, 'rh.highlevelvisual.mgz')

# load the lh mask
maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

maskdata = np.hstack((maskdata_lh, maskdata_rh))

ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}

roi_names = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']

# sessions
n_sessions = 40

# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

# extract conditions
conditions = get_conditions(nsd_dir, sub, n_sessions)

# we also need to reshape conditions to be ntrials x 1
conditions = np.asarray(conditions).ravel()

# then we find the valid trials for which we do have 3 repetitions.
conditions_bool = [
    True if np.sum(conditions == x) == 3 else False for x in conditions]

conditions_sampled = conditions[conditions_bool]

# find the subject's unique condition list (sample pool)
sample = np.unique(conditions[conditions_bool])

betas_file = os.path.join(
    outpath, f'{sub}_betas_list_{targetspace}.npy'
)
betas_mean_file = os.path.join(
        outpath, f'{sub}_betas_list_{targetspace}_averaged.npy'
)

if not os.path.exists(betas_mean_file):
    # get betas
    betas_mean = get_betas(
        nsd_dir,
        sub,
        n_sessions,
        targetspace=targetspace,
    )
    print(f'concatenating betas for {sub}')
    betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

    print(f'averaging betas for {sub}')
    betas_mean = average_over_conditions(
        betas_mean,
        conditions,
        conditions_sampled,
    ).astype(np.float32)

    # print
    print(f'saving condition averaged betas for {sub}')
    np.save(betas_mean_file, betas_mean)

else:
    print(f'loading betas for {sub}')
    betas_mean = np.load(betas_mean_file, allow_pickle=True)


# print
print(f'saving condition list for {sub}')
np.save(
        os.path.join(
            outpath, f'{sub}_condition_list.npy'
        ),
        conditions_sampled
    )

# save the subject's full ROI RDMs
for roi in range(1, 6):
    mask_name = ROIS[roi]

    rdm_file = os.path.join(
        outpath, f'{sub}_{mask_name}_fullrdm_correlation.npy'
    )

    if not os.path.exists(rdm_file):

        # logical array of mask vertices
        vs_mask = maskdata == roi
        print(f'working on ROI: {mask_name}')

        masked_betas = betas_mean[vs_mask, :]

        good_vox = [
            True if np.sum(
                np.isnan(x)
                ) == 0 else False for x in masked_betas]

        if np.sum(good_vox) != len(good_vox):
            print(f'found some NaN for ROI: {mask_name} - {sub}')

        masked_betas = masked_betas[good_vox, :]

        # prepare for correlation distance
        X = masked_betas.T

        print(f'computing RDM for roi: {mask_name}')
        start_time = time.time()
        rdm = pdist(X, metric='correlation')

        if np.any(np.isnan(rdm)):
            raise ValueError

        elapsed_time = time.time() - start_time
        print(
            'elapsedtime: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )
        print(f'saving full rdm for {mask_name} : {sub}')
        np.save(
            rdm_file,
            rdm
        )
