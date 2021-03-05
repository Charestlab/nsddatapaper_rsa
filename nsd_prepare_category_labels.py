import os.path as op
import numpy as np
from nsd_access import NSDAccess
from nsd_get_data import get_conditions

# number of sessions
n_sessions = 40

# number of subjects
n_subjects = 8

# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

# setup some directories
base_dir = op.join('/media')
nsd_dir = op.join(base_dir, 'charesti-start', 'data', 'NSD')
proj_dir = op.join(base_dir, 'charesti-start', 'projects', 'NSD')
betas_dir = op.join(proj_dir, 'rsa')

# prepare outputs
outpath = op.join(proj_dir, 'rsa')

# conditions
save_stim = op.join(outpath, 'all_subs_stims_full.npy')
if not op.exists(save_stim):
    # get unique list of conditions
    conditions_all = []
    for sub in subs:
        # extract conditions data
        conditions = get_conditions(nsd_dir, sub, n_sessions)
        # we also need to reshape conditions to be ntrials x 1
        conditions = np.asarray(conditions).ravel()
        # append across subjects
        conditions_all.append(conditions)

    # this is the 73K format condition name for all 213 000 trials
    # across subjects
    conditions_all = np.concatenate(conditions_all, axis=0)

    # this is as above, but without repeats.
    # useful later when we average across trials for each cond
    conditions = np.unique(conditions_all)

    # save the conditions
    np.save(save_stim, conditions)
else:

    # get the condition list
    conditions = np.load(op.join(
        betas_dir, 'all_subs_stims_full.npy'
    ), allow_pickle=True)


# now that we have all the conditions, let's read in the
# category labels.
nsda = NSDAccess(nsd_dir)

# in nsda we have a module that extracts the categories
# from the annotation files provided with MS coco.
categories = nsda.read_image_coco_category(
        conditions-1, njobs=2
    )

# and save them
np.save(op.join(betas_dir, 'all_stims_category_labels.npy'), categories)
