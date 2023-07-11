# self-consistent-expression-recovery-machine (SERM)
SERM is a data-driven gene expression recovery framework. Using deep learning, SERM first learns from a subset of the noisy gene expression data to estimate the underlying data distribution. SERM then recovers the overall gene expression data by analytically imposing a self- consistency on the gene expression matrix, thus ensuring that the expression levels are similarly distributed in different parts of the matrix. SERM performs much better (in most cases >20% improvement than State-of-the-art) and is computationally at least 10 times faster than other analytical techniques. Here is an exmaple of its extra-ordinary performance:

![image](https://github.com/xinglab-ai/self-consistent-expression-recovery-machine/blob/main/im1.png)
Figure. Percent improvement in mean Pearson coefficient of the imputed data (in comparison to the observed data) by eight different techniques for cellular taxonomy  dataset. The sampling efficiencies (0.1%-10%) to create the observed data are shown in x-axis.

# How to use SERM?

## Input data and parameters
The gene expression data should be in matrix format (rows are cells and collumns are genes). There are a few parameters: 1) ROI length and width: SERM divides the whole matrix into a number of ROIs. The user can choose the size of the ROI that would be used by SERM. Good values for ROI height can be: cell number divided by 4 or 8 and ROI width can be gene number divided by 4 or 8. 2) Percent overlap: This parameter dictates how much overlap will be between two consecutive ROIs. The user can choose any value between 0.01 to 0.99. However, for small overlap, the consecutive windows would be disconnected and for large window size, more computation will be necessary. A value between 0.25 or 0.75 works well with SERM. The requirements of different packages in Python are described in requirements.txt. 

## About parameter settings:
SERM works well for a large range of parameters. Thus, even without careful settings, it provides much better results than traditional techniques.

# Where do I get all the data used in the manuscript?
Please download all the reference and observed data for all six datasets from the following Google Drive link. 

https://drive.google.com/drive/folders/1w72k3fCQlS2UtGg2TdY8CihTwavEmz89?usp=sharing

Please refer to our Nature Communications paper, if you use the data/code in your research.

# Citation:

Islam, M.T., Wang, JY., Ren, H. et al. Leveraging data-driven self-consistency for high-fidelity gene expression recovery. Nat Commun 13, 7142 (2022). https://doi.org/10.1038/s41467-022-34595-w

# Code Ocean capsule:

You can run our reproducible code ocean capsule, where you just need to click once to obtain the results. Link to the capsule: https://doi.org/10.24433/CO.7874136.v1

# Sample data

To run the example code below, you will need to download the required data files. You can download them from [here](https://github.com/xinglab-ai/self-consistent-expression-recovery-machine/tree/main/demo/data).

# Example code:

```python
# Import all the necessary Python packages

import scipy.io as sio
import serm as srm
import random
import time

# Load the reference data, the observed data, and the data labels
# The raw data is from the work of Baryawno et al. 
# Reference: Baryawno, Ninib, et al. "A cellular taxonomy 
# of the bone marrow stroma in homeostasis and leukemia." Cell 177.7 (2019): 1915-1932.
# The data from the authors contains 23092 cells and 27998 genes. 
# The raw data is added in 'data' folder of this capsule. Following Huang et al 
# (Nature Methods 15, 539–542 (2018)), we choose 12,162 cells and 2,422 genes with 
# high expression to create a reference dataset.
# 
# The following two code lines read the reference data 
a=sio.loadmat('demo/data/data_reference.mat')
dataRef=a['data_reference']

# The observed data were generated from the reference data by simulating efficiency loss that introduces zeros 
# following the work of Huang et al.
# The following three code lines were used from Huang et al to create observed data with 10%
# sampling efficiency: (detail code (demo_creation_observedData.R) is also added in the 'code' folder
# of this capsule)
# alpha <- rgamma(ncol(data.filt), 10, 100)
# data_observed <- t(apply(sweep(data.filt, 2, alpha, "*"), 1, function(data)
#  rpois(length(data), data)))

# The following two code lines read the observed data for 10% sampling efficiency
a=sio.loadmat('demo/data/data_observed_10perc.mat')
dataObs=a['data_observed']

# Load data labels 
a=sio.loadmat('demo/data/data_label.mat')
dataLabel=a['data_label']

# Selection of the size of the region of interest (ROI)
# We choose ROI size in such a way that the expression matrix
# is divided into 4 (i.e. 2 by 2) ROIs. However, other ROI sizes are also
# fine as SERM is generally very robust against the ROI size. 
ROIsize = [6100,1250]
# Selection of percent of overlap between successive ROIs. The default value is 50%, but other 
# values can be chosen
percOL = 0.5
t=time.time() # For keeping record of computational time
random.seed(0) # Setting the random number generator seed to 0. 
# Run SERM
out_SERM = srm.serm(dataObs,ROIsize,percOL,randomize=False) # If you want to randomize the rows and columns
# of the data before applying SERM, please choose randomize=True
print('Required time:',str(time.time()-t),'second')
```

# Results:

![image](https://github.com/xinglab-ai/self-consistent-expression-recovery-machine/blob/main/im1.png)
Figure. Analysis of simulated scRNA-seq data with 5 classes. The histograms of the reference data, observed data (1% sampling efficiency), and imputed data by MAGIC, mcImpute, and SERM are shown in the first row of (a). Visualization of reference, observed,  and imputed data by t-SNE and UMAP are shown in the second and third rows, respectively. t-SNE and UMAP results from SERM imputed data are much better in separating the classes, whereas MAGIC degrades the data as a result of imputation. The clustering accuracy and cluster quality indices for UMAP visualizations of imputed data from different methods are shown in (b). Data are presented as mean values +/- standard deviation (SD). Error bars represent the standard deviation of the indices for n=1000 different initializations of k-means clustering.
