# nasdatapaper_rsa
rsa analyses for the data paper

this is a set of scripts that will process nsd fmri data
and mask in a region of interest.

first, you need to compute the category labels with

nsd_prepare_category_labels.py

second, you need to prepare the masked betas in rois
along the ventral stream with

nsd_prepare_rois_rdms.py

finally, you can plot the TSNE or MDS with
nsd_plot_tsne.py