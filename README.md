# nsddatapaper_rsa

## rsa analyses for the data paper

this is a set of scripts that will process nsd fmri data,
mask in a set of regions of interest along the ventral stream,
compute representational dissimilarity matrices from all pairs
condition activity patterns, and finally plot the TSNE 
two-dimensional solutions for those RDMs.

### to install simply clone this repo, cd to it's directory and 

```bash
python setupy.py develop
```

this will install the package and all other [required packages](requirements.txt).


### additionally, you may need to install Tomas Knapen's nsd_access

```bash
pip install git+https://github.com/tknapen/nsd_access.git
```

nsd_access is a convenient tool for quickly and easily accessing data from the 
Natural Scenes Dataset. For more details and tutorials see the repo here: https://github.com/tknapen/nsd_access


### first, you need to compute the category labels with

```bash
python nsd_prepare_category_labels.py
```

### second, you need to prepare the masked betas in rois along the ventral stream with

```bash
python nsd_prepare_rois_rdms.py 0
```

this will prepare RDMs for all the ROIs for subject 0, which is in fact
NSD subj01.


### finally, you can plot the TSNE or MDS with

```bash
python nsd_plot_tsne.py 0
```

this will create a series of plots from the paper, for subj01.