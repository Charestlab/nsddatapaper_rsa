"""[nds_get_data]

    utilies for nsd
"""
import numpy as np
import nibabel as nb
from scipy.stats import zscore
from nsd_access import NSDAccess
import os


def get_conditions(nsd_dir, sub, n_sessions):
    """conditions = get_conditions(nsd_dir, sub, n_sessions)

    Arguments:
    __________

        nsd_dir (os.path): absolute path to the NSD data folder.

        sub (string): subject identifier (e.g. subj01)

        n_sessions (int): the number of sessions to fetch data from

    Returns:
    __________

        [type]: [description]
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    # read behaviour files for current subj
    conditions = []

    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses+1
        print(f'\t\tsub: {sub} fetching condition trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = np.asarray(
            nsda.read_behavior(subject=sub, session_index=ses_i)['73KID'])

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return conditions


def get_betas(nsd_dir, sub, n_sessions, mask=None, targetspace='func1pt8mm'):
    """ betas = get_betas(nsd_dir, sub, n_sessions, mask, targatspace)

    Arguments:
    ___________

        nsd_dir (os.path): absolute path to the NSD data folder.

        sub (string): subject identifier (e.g. subj01)

        n_sessions (int): the number of sessions to fetch data from

        mask (bool or index, optional): logical mask (e.g. a specific roi)

        targetspace (str, optional): Data preparation space.
            Defaults to 'func1pt8mm'.

    Returns:
    __________

        array: numpy array of betas with shape features x conditioons
    """

    # initiate nsda
    nsda = NSDAccess(nsd_dir)

    data_folder = os.path.join(
        nsda.nsddata_betas_folder,
        sub,
        targetspace,
        'betas_fithrf_GLMdenoise_RR')

    betas = []
    # loop over sessions
    # trial_index=0
    for ses in range(n_sessions):
        ses_i = ses+1
        si_str = str(ses_i).zfill(2)

        # sess_slice = slice(trial_index, trial_index+750)
        print(f'\t\tsub: {sub} fetching betas for trials in session: {ses_i}')

        # we only want to keep the shared_1000
        this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)

        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses['73KID'])

        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:

            if targetspace == 'fsaverage':
                conaxis = 1

                # load lh
                img_lh = nb.load(
                        os.path.join(
                            data_folder,
                            f'lh.betas_session{si_str}.mgz'
                            )
                        ).get_data().squeeze()

                # load rh
                img_rh = nb.load(
                        os.path.join(
                            data_folder,
                            f'rh.betas_session{si_str}.mgz'
                            )
                        ).get_data().squeeze()

                # concatenate
                all_verts = np.vstack((img_lh, img_rh))

                # mask
                if mask is not None:
                    tmp = zscore(all_verts, axis=conaxis).astype(np.float32)

                    # you may want to get several ROIs from a list of ROIs at
                    # once
                    if type(mask) == list:
                        masked_betas = []
                        for mask_is in mask:
                            tmp2 = tmp[mask_is, :]
                            # check for nans
                            # good = np.any(np.isfinite(tmp2), axis=1)
                            masked_betas.append(tmp2)
                    else:
                        tmp2 = tmp[mask_is, :]
                        masked_betas = tmp2

                    betas.append(masked_betas)
                else:
                    betas.append(
                        (zscore(
                            all_verts,
                            axis=conaxis)).astype(np.float32)
                        )
            else:
                conaxis = 1
                img = nb.load(
                    os.path.join(data_folder, f'betas_session{si_str}.nii.gz'))

                if mask is not None:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)[mask, :]*300).astype(np.int16)
                        )
                else:
                    betas.append(
                        (zscore(
                            np.asarray(
                                img.dataobj),
                            axis=conaxis)*300).astype(np.int16)
                        )

    return betas


def get_labels(sub, betas_dir, nsd_dir, condition_list):
    """labels = get_labels(sub, betas_dir, nsd_dir, condition_list)

    Arguments:
    __________

        sub (str): NSD subject identifier (e.g. subj01)

        betas_dir (os.path): absolute path where the betas are saved

        nsd_dir (os.path): absolute path where the NSD data lives

        condition_list (int or list): lits of nsd conditions

    Returns:
    __________

        [array]: one hot like vector of categories.
    """

    nsda = NSDAccess(nsd_dir)

    # get the categories
    labels = np.load(os.path.join(
        betas_dir, 'all_stims_category_labels.npy'
    ), allow_pickle=True)

    label_file = os.path.join(betas_dir, f'{sub}_sample_labels.npy')

    if not os.path.exists(label_file):
        print('computing category labels')
        # convert category names to binary vectors
        # scrape all available text labels:
        flat_labels = [item for sublist in labels for item in sublist]
        all_labels = sorted(list(set(flat_labels)))
        num_labels = len(all_labels)

        # index to text label
        labels_to_indices = {all_labels[i]: i for i in range(len(all_labels))}
        # text label to index

        # get the specific labels for the condition list and binarise
        categories = nsda.read_image_coco_category(
            condition_list
        )

        bin_vectors = []
        for category in categories:
            bin_vector = np.zeros(num_labels)
            idx_vector = [labels_to_indices[x] for x in category]
            bin_vector[idx_vector] = 1
            bin_vectors.append(bin_vector)

        label_matrix = np.asarray(bin_vectors)

        print(f'saving serialised label matrix to:\n\t {label_file}')
        np.save(label_file, label_matrix)

    else:
        print(f'loading serialised label matrix from:\n\t {label_file}')
        label_matrix = np.load(label_file, allow_pickle=True)

    return label_matrix
