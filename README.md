# nasdatapaper_rsa
rsa analyses for the data paper

this is a set of scripts that will process nsd fmri data,
mask in a set of regions of interest along the ventral stream,
compute representational dissimilarity matrices from all pairs
condition activity patterns, and finally plot the TSNE 
two-dimensional solutions for those RDMs.

first, you need to compute the category labels with

nsd_prepare_category_labels.py

second, you need to prepare the masked betas in rois
along the ventral stream with

nsd_prepare_rois_rdms.py

finally, you can plot the TSNE or MDS with
nsd_plot_tsne.py