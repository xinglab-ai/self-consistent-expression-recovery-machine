import scipy
import scipy.io as sio
from scipy.interpolate import interp1d
import numpy as np
from .sparseae import *
from scipy.optimize import curve_fit
import os
from functools import partial
import random
from numba import jit


def serm(data,ROIsize=[500,100], percOL=0.5, randomize=True):
    # Applies serm technique on gene expression data for recovering the true
    # expression values
    #
    # Inputs:
    #  x: gene expression matric. Rows denote the cells and columns denote the
    # genes
    #  ROIsize: Size of the ROIs for imputation operation
    #  percOL: Percent overlap between successive ROIs
    #
    # Return:
    # the recovered gene expression matrix
    # Set the number of data points from where the distribution of denoised expression will be
    # learnt
    numLearningDataPoints=2000
    random.seed(0) # Setting the random number generator seed to 0. 
    sz=data.shape
    if (randomize==True):
        Xcopy=np.copy(data)        
        XcopyF=Xcopy.flatten() # Flatten the data matrix for randomization
        Xc, p = shuffle(XcopyF) # Keep record of the randomization
        X=np.reshape(Xc, (sz[0], sz[1])) # Convert the randomized vector to
        # a matrix for SERM operation  
    else: 
        X=data
    
    # Data rescaling from 0 to 1
    minX = np.min(X)
    maxX = np.max(X)    
    Xre = (X - minX) / (maxX - minX)    
    
    
    # Selecting the points for learning distribution of denoised expression values
    if (sz[0]>numLearningDataPoints):
        idx=random.sample(np.arange(sz[0]).tolist(), numLearningDataPoints)
        dataPartLearning=Xre[idx,:]
        dataPartLearning = dataPartLearning/dataPartLearning.max()
    else:
        dataPartLearning = Xre/Xre.max()
    
    # Training an autoencoder for reconstructing the denoised expression values
    denoisedData = trainAutoencoder(dataPartLearning, reduced_Dim=2, maxEPOCH=20,batchSize=64) 

    # Find the distribution of denoised data
    idealDistributionName,paramX = findDistribution(denoisedData)
    
    # Perform histogram equalization using the leant distribution
    if (percOL==0): # If no percent overlap, apply discrete histogram equalization        
        xTileNum=np.ceil(sz[0]/ROIsize[0])
        yTileNum=np.ceil(sz[1]/ROIsize[1])
        X_adapthisteq = adapthisteq(Xre,numTiles=[int(xTileNum),int(yTileNum)],clipLimit=0.0,numBins=256,
                                    distribution=idealDistributionName,alpha=paramX)
    else:
        X_adapthisteq = recovery(Xre, idealDistributionName, paramX, ROIsize, percOL)
    
    if (randomize==True):        
        X_adapthisteqX, _ = shuffle(X_adapthisteq.flatten(),permutation=p[::-1])
        # Use the randomization index to put back the original matrix
        X_adapthisteq=np.reshape(X_adapthisteqX, (sz[0], sz[1]))        
        
    Xk = X_adapthisteq - np.min(X_adapthisteq) # Rescale back 
    sermOut=minX + (maxX - minX) * Xk
    sermOut=sermOut*(np.sum(data)/np.sum(sermOut))
    return sermOut


def computeResiduals(x,y,f):
    # computes residuals of fitted function, y_hat - y = f(x) - y
    #
    # Inputs:
    #  x: independent variables
    #  y: observed dependent variabls
    #  f: fitted function, with only one parameter x
    #
    # Return:
    #  f(x) - y
    return y - f(x)

def computeRMSE(x,y,f):
    # computes RMSE of fitted function, sqrt((y_hat - y)^2/n)
    #
    # Inputs:
    #  x: independent variables
    #  y: observed dependent variabls
    #  f: fitted function, with only one parameter x
    #
    # Return:
    #  sqrt((y_hat - y)^2/n) = sqrt((f(x) - y)^2/n)
    residuals = computeResiduals(x,y,f)
    return np.sqrt(np.mean(residuals**2))
    
def gauss_func(x, p, c):
    # Gaussian Distribution
    # (1/(sqrt(pi)*c))*exp(-(x/c)^2)
    return (p/(np.sqrt(np.pi*c)))*np.exp(-(x/c)**2)

def exp_func(x, a, b):
    # Exponential Distribution
    # a*exp(-(b*x)). a is used here to take care of any scaling in the fitted 
    # data. Thus, a=M*b, where M is a scaling factor.  This also helps in 
    # obtaining better parameter estimates from noisy data. 
    return a*np.exp(-(b*x))

def ray_func(x, p, c):
    # Rayleigh distribution
    # ((2*x)/c^2)*exp(-(x/c)^2)
    return ((p*2*x)/(c**2))*np.exp(-(x/c)**2)

def findDistribution(X):
    # learns the distribution of the denoised data 
    #
    # Inputs:
    #  X: the data that is transformed by a Sparse Autoencoder,
    #   each row is a data sample
    # 
    # Return:
    #  idealDistribution: Distribution of the denoised expression data
    Xrec = X.T
    tempDist = np.arange(0.01,1.01,0.01)
    NBin = tempDist.shape[0]
    Xn = (Xrec-np.min(Xrec)) / (np.max(Xrec)-np.min(Xrec))
    [N,edges] = np.histogram(Xn,bins=NBin,density=True)
    probs = N * 0.01     
    tempDist=tempDist[1:]
    probs=probs[1:]
    
    gaussCoeff,_ = curve_fit(gauss_func, tempDist, probs,
                             bounds=([0, 0.01],[np.inf, 1]), p0=[1,0.2])
    gaussEqn = partial(gauss_func, p=gaussCoeff[0], c=gaussCoeff[1])

    min_rmse = computeRMSE(tempDist, probs, gaussEqn)
    min_dist='gaussian'
    min_param=gaussCoeff
    residuals = computeResiduals(tempDist, probs, gaussEqn)
    
    rayCoeff,_ = curve_fit(ray_func, tempDist, probs,
                           bounds=([0., 0.01],[np.inf, 1]), p0=[1, 0.2])
    rayEqn = partial(ray_func, p=rayCoeff[0], c=rayCoeff[1])

    if computeRMSE(tempDist, probs, rayEqn) < min_rmse:
        min_rmse = computeRMSE(tempDist, probs, rayEqn)
        residuals = computeResiduals(tempDist, probs, rayEqn)
        min_dist='rayleigh'
        min_param=rayCoeff

    expCoeff,_ = curve_fit(exp_func, tempDist, probs,
                           bounds=([0., 5.],[np.inf, 20.]), p0=[1,5])
    expEqn = partial(exp_func, a=expCoeff[0], b=expCoeff[1])

    if computeRMSE(tempDist, probs, expEqn) < min_rmse:
        min_rmse = computeRMSE(tempDist, probs, expEqn)
        residuals = computeResiduals(tempDist, probs, expEqn)
        min_dist='exponential'
        min_param=expCoeff

    print('The selected distribution:', str(min_dist), 'and the optimized value of distribution parameter:', str(min_param[1]))
      
    return min_dist,min_param[1]

def shuffle(A,axis=0,permutation=None):
    # Shuffles the rows of the data randomly and keeps the record of the 
    # shuffling index
    A = np.swapaxes(A,0,axis)
    if permutation is None:
        permutation = np.random.permutation(len(A))
    temp = np.copy(A[permutation[0]])
    for i in range(len(A)-1):
        A[permutation[i]] = A[permutation[i+1]]
    A[permutation[-1]] = temp
    A = np.swapaxes(A,0,axis)
    return A, permutation

def recovery(data, idealDistributionName, paramX, ROIsize, percOL):
    # Produces a recovered gene expression matrix from expressions with dropout
    # and ideal data distribution.
    #
    # Inputs:
    #  data: double matrix, gene expression data 
    #   in tabular format, i.e., rows denote cells and columns denote the genes. 
    #  idealDistribution: double vector 
    #   Ideal distribution of the data without dropout
    #  ROIsize: Height and width of the ROIs used in SERM to recover the
    #   expression values
    #  percOL: Percent overlap between two consecutive ROIs. The value should
    #   between 0 and 1. 
    #
    # Outputs:
    #  recoveredMatrix: SERM-recovered gene expression matrix
   
    ROIsize = np.array(ROIsize)

    rs,cs = data.shape

    numROIr = np.ceil(rs/ROIsize[0])
    numROIc = np.ceil(cs/ROIsize[1])
    rsEdit = numROIr*ROIsize[0];
    csEdit = numROIc*ROIsize[1];

    if (rsEdit-rs)%2 == 1:
        rsEdit += 1

    if (csEdit-cs)%2 == 1:
        csEdit += 1

    r_pad = int((rsEdit-rs)/2)
    c_pad = int((csEdit-cs)/2)
    Xpad = np.pad(data,[(r_pad,r_pad),(c_pad, c_pad)],
                  mode='constant', constant_values=0)

    rsp,csp = Xpad.shape
    OLsize = np.ceil(ROIsize*percOL)

    numROIrp = np.ceil(rsp/OLsize[0])
    numROIcp = np.ceil(csp/OLsize[1])

    rsEdit2 = numROIrp*OLsize[0]
    csEdit2 = numROIcp*OLsize[1]

    if (rsEdit2-rsEdit)%2 == 1:
        rsEdit2 = rsEdit2+1

    if (csEdit2-csEdit)%2 == 1:
        csEdit2 = csEdit2+1

    r_pad2 = int((rsEdit2-rsEdit)/2)
    c_pad2 = int((csEdit2-csEdit)/2)
    Xpad2 = np.pad(Xpad,[(r_pad2,r_pad2),(c_pad2, c_pad2)],
                  mode='constant', constant_values=0)

    X = Xpad2

    M = 50 
    meanX = np.mean(X)
    X[X>M*meanX] = M*meanX

    minX = np.min(X)
    maxX = np.max(X)

    Xre = (X-np.min(X)) / (np.max(X)-np.min(X))

    sx,sy = X.shape
    overlappixelX = int(np.ceil(percOL*ROIsize[0]))
    overlappixelY = int(np.ceil(percOL*ROIsize[1]))
    numItX2 = np.ceil(sx/overlappixelX)
    numItY2 = np.ceil(sy/overlappixelY)

    numItX = int(np.ceil(sx/overlappixelX)-(1/percOL)+1)
    numItY = int(np.ceil(sy/overlappixelY)-(1/percOL)+1)


    Xpad_r = int(np.round((numItX2*overlappixelX-sx)/2))
    Xpad_c = int(np.round((numItY2*overlappixelY-sy)/2))

    Xpad = np.pad(Xre,[(Xpad_r,Xpad_r),(Xpad_c,Xpad_c)],
                  mode='constant', constant_values=0)

    XpadSave = np.copy(Xpad)

    for i in range(numItX):
        for j in range(numItY):
            Xtake = Xpad[(i*overlappixelX):(i*overlappixelX+ROIsize[0]),
                         (j*overlappixelY):(j*overlappixelY+ROIsize[1])]
            
            Xh = adapthisteq(Xtake,numTiles=[2,2],clipLimit=0.0,numBins=256,
                                        distribution=idealDistributionName,alpha=paramX)
            
            #Xh = _match_hist(Xtake,idealDistribution)
            XpadSave[(i*overlappixelX):(i*overlappixelX+ROIsize[0]),
                     (j*overlappixelY):(j*overlappixelY+ROIsize[1])] = Xh
            

    Xhist = XpadSave[int(np.round((numItX2*overlappixelX-sx)/2)):\
                     XpadSave.shape[0]-int(np.round((numItX2*overlappixelX-sx)/2)),
                     int(np.round((numItY2*overlappixelY-sy)/2)):\
                     XpadSave.shape[1]-int(np.round((numItY2*overlappixelY-sy)/2))]
    Xk = Xhist-np.min(Xhist)
    Y = minX+(maxX-minX)*Xk
    Ypad2 = Y[int((rsEdit2-rsEdit)/2):Y.shape[0]-int((rsEdit2-rsEdit)/2),
              int((csEdit2-csEdit)/2):Y.shape[1]-int((csEdit2-csEdit)/2)]
    recoveredMatrix = Ypad2[int((rsEdit-rs)/2):Ypad2.shape[0]-int((rsEdit-rs)/2),
                            int((csEdit-cs)/2):Ypad2.shape[1]-int((csEdit-cs)/2)]
    
    return recoveredMatrix

def set_axis_style(ax, labels):
    # A helper function for violin plots
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    
def adapthisteq(I, numTiles=[2,2], clipLimit=0.0, numBins=256, distribution='rayleigh', alpha=0.2):
    # ADAPTHISTEQ Contrast-limited Adaptive Histogram Equalization (CLAHE).
    # ADAPTHISTEQ enhances the contrast of images by transforming the
    # values in the intensity image I.  Unlike HISTEQ, it operates on small
    # data regions (tiles), rather than the entire image. Each tile's
    # contrast is enhanced, so that the histogram of the output region
    # approximately matches the specified histogram. The neighboring tiles
    # are then combined using bilinear interpolation in order to eliminate
    # artificially induced boundaries.  The contrast, especially
    # in homogeneous areas, can be limited in order to avoid amplifying the
    # noise which might be present in the image.
    # Inputs:
    #  I: Image that CLAHE is performed on
    #  numTiles: List of number of tiles for rows and columns.
    #  clipLimit:  Real scalar from 0 to 1. 'ClipLimit' limits contrast 
    #   enhancement. Higher numbers result in more contrast.
    #  numBins:  Positive integer scalar. Sets number of bins for the 
    #   histogram used in building a contrast enhancing transformation. 
    #   Higher values result in greater dynamic range at the cost of slower 
    #   processing speed.
    #  distribution:  Distribution can be one of three strings or character
    #   vectors: 'uniform', 'rayleigh', 'exponential'. Sets desired histogram 
    #   shape for the image tiles, by specifying a distribution type.
    #  alpha:  Nonnegative real scalar. 'Alpha' is a distribution parameter, 
    #   which can be supplied  when 'Dist' is set to either 'rayleigh' or 
    #   'exponential'.
    #
    # Outputs:
    #  Image after CLAHE
    I, pad = padImage(I, numTiles)
    dimTile = (int(I.shape[0] / numTiles[0]),int(I.shape[1] / numTiles[1]))
    
    numPixInTile = np.prod(dimTile)
    minClipLimit = int(np.ceil(numPixInTile / numBins))
    clipLimit = minClipLimit + np.round(clipLimit * (numPixInTile - minClipLimit))
    
    fullRange = [0,1]
#     fullRange = [np.min(I), np.max(I)]
    selectedRange = fullRange
    
    tileMappings = makeTileMappings(I, numTiles, dimTile, numBins, clipLimit,
                                    selectedRange, fullRange, distribution, alpha)
    
    # Synthesize the output image based on the individual tile mappings.
    out = makeClaheImage(I, tileMappings, numTiles, selectedRange, numBins, dimTile)
    
    return out[pad[0][0]: I.shape[0]-pad[0][1], pad[1][0]: I.shape[1]-pad[1][1]]


def padImage(I, numTiles, return_pad=True):
    # Check if the image needs to be padded; pad if necessary;
    # padding occurs if any dimension of a single tile is an odd number
    # and/or when image dimensions are not divisible by the selected
    # number of tiles
    # Inputs:
    #  I: Image to be padded
    #  numTiles: List of number of tiles for rows and columns
    #  return_pad: whether return the pad number of each dimension
    #
    # Outputs:
    #  Padded Image (, and padded number of rows and columns if return_pad=True)
    [row_img, col_img] = I.shape
    pad_rows = row_img % int(numTiles[0])
    pad_cols = col_img % int(numTiles[1])
    
    if (pad_rows + row_img) % (2 * int(numTiles[0])) != 0:
        pad_rows += numTiles[0]
    if (pad_cols + col_img) % (2 *int(numTiles[0])) != 0:
        pad_cols += numTiles[1]
    
    pad_rows = (int(pad_rows/2), pad_rows - int(pad_rows/2))
    pad_cols = (int(pad_cols/2), pad_cols - int(pad_cols/2))
    
    if return_pad:
        return np.pad(I, (pad_rows, pad_cols), mode='symmetric'), (pad_rows, pad_cols)
    else:
        return np.pad(I, (pad_rows, pad_cols), mode='symmetric')

@jit(nopython=True)
def clipHistogram(imgHist, clipLimit, numBins):
    # This function clips the histogram according to the clipLimit and
    # redistributes clipped pixels across bins below the clipLimit

    # total number of pixels overflowing clip limit in each bin
    totalExcess = np.sum(np.maximum(imgHist - clipLimit,0))
    # clip the histogram and redistribute the excess pixels in each bin
    avgBinIncr = int(totalExcess/numBins)
    # bins larger than this will be set to clipLimit
    upperLimit = clipLimit - avgBinIncr
    
    # this loop should speed up the operation by putting multiple pixels
    # into the "obvious" places first
    for k in range(numBins):
        if imgHist[k] > clipLimit:
            imgHist[k] = clipLimit
        else:
            #  high bin count
            if imgHist[k] > upperLimit:
                totalExcess = totalExcess - (clipLimit - imgHist[k])
                imgHist[k] = clipLimit
            else:
                totalExcess = totalExcess - avgBinIncr
                imgHist[k] = imgHist[k] + avgBinIncr
    
    # this loops redistributes the remaining pixels, one pixel at a time
    k = 0
    while totalExcess != 0:
        # keep increasing the step as fewer and fewer pixels remain for 
        # the redistribution (spread them evenly)
        stepSize = max(int(numBins / totalExcess), 1)
        for m in range(k, numBins, stepSize):
            if imgHist[m] < clipLimit:
                imgHist[m] = imgHist[m] + 1
                # reduce excess
                totalExcess = totalExcess - 1
                if totalExcess == 0:
                    break
        # prevent from always placing the pixels in bin #1
        k += 1
        # start over if numBins was reached
        if k > numBins - 1:
            k = 0

    return imgHist

def grayxform(I, aLut):
    # Python version of GRAYXFORMMEX, map I to aLut
    # Inputs:    
    #  I: image
    #  aLut: look-up table

    max_idx = len(aLut) - 1
    val = np.copy(I)
    val[val < 0] = 0
    val[val > 1] = 1
    indexes = np.int32(val * max_idx + 0.5)
    return aLut[indexes]
    
def rescale(A):
    # rescale a vector from 0 to 1
    return (A-np.min(A))/(np.max(A) - np.min(A))

def makeMapping(imgHist, selectedRange, fullRange, numPixInTile,
                distribution, alpha):
    # Calculate the equalized lookup table (mapping) based on cumulating the input
    # histogram.  Note: lookup table is rescaled in the selectedRange [Min..Max].

    histSum = np.cumsum(imgHist)
    valSpread  = selectedRange[1] - selectedRange[0]
    
    if distribution == 'gaussian':
        hconst = np.sqrt(2) * alpha
        vmax = 0.5*(1+scipy.special.erf(1. / hconst))
        val = vmax * (histSum / numPixInTile)
        val[val>=1] = 1 - np.finfo(float).eps
        temp=hconst*scipy.special.erfinv(2*val-1)
        temp = rescale(temp)
        mapping = np.minimum(selectedRange[0] + temp * valSpread, selectedRange[1])
    elif distribution == 'rayleigh':
        hconst = 2 * (alpha ** 2)
        vmax = 1. - np.exp(-1. / hconst)
        val = vmax * (histSum / numPixInTile)
        val[val>=1] = 1 - np.finfo(float).eps
        temp = np.sqrt(-hconst * np.log(1 - val))
        mapping = np.minimum(selectedRange[0] + temp * valSpread, selectedRange[1])
    elif distribution == 'exponential':
        vmax = 1 - np.exp(-alpha)
        val = (vmax * histSum / numPixInTile)
        val[val>=1] = 1 - np.finfo(float).eps
        temp = -1 / alpha * np.log(1 - val)
        mapping = np.minimum(selectedRange[0] + temp * valSpread, selectedRange[1])
    else:
        pass # Not implemented
    # rescale the result to be between 0 and 1 for later use by the grayxform
    # private mex function
    mapping = mapping / fullRange[1]
    return mapping

def makeTileMappings(I, numTiles, dimTile, numBins, clipLimit,
                     selectedRange, fullRange, distribution, alpha):
    
    numPixInTile = np.prod(dimTile)
    tileMappings = {}
    
    # extract and process each tile
    imgCol = 0
    for col in range(numTiles[1]):
        imgRow = 0
        for row in range(numTiles[0]):
            tile = I[imgRow:imgRow+dimTile[0], imgCol:imgCol+dimTile[1]]
            tileHist, _ = np.histogram(tile, numBins, (fullRange[0],fullRange[1]))
            tileHist = clipHistogram(tileHist, clipLimit, numBins)
            tileMapping = makeMapping(tileHist, selectedRange, fullRange,
                                      numPixInTile, distribution, alpha)
            # assemble individual tile mappings by storing them in a cell array
            tileMappings[(row,col)] = tileMapping
            imgRow = imgRow + dimTile[0]
        imgCol = imgCol + dimTile[1]
    return tileMappings

def makeClaheImage(I, tileMappings, numTiles, selectedRange, numBins, dimTile):
    # This function interpolates between neighboring tile mappings to produce a
    # new mapping in order to remove artificially induced tile borders.
    # Otherwise, these borders would become quite visible.  The resulting
    # mapping is applied to the input image thus producing a CLAHE processed
    # image.

    # initialize the output image to zeros (preserve the class of the input image)
    claheI = np.zeros(I.shape)
    
    # compute the LUT for looking up original image values in the tile mappings,
    # which we created earlier
    if selectedRange[1] == 1 and selectedRange[0] == 0:
        # remap from 0..1 to 0..numBins-1
        if numBins != 1:
            binStep = 1 / (numBins - 1)
            start = int(np.ceil(selectedRange[0] / binStep))
            stop = int(selectedRange[1] / binStep)
            aLut = np.arange(0, 1 + 1/(stop - start), 1/(stop - start))
        else:
            # in case someone specifies numBins = 1, which is just silly
            aLut = np.array([0,])
    else:
        aLut = np.arange(selectedRange[0], selectedRange[1] + 1) - selectedRange[0]
        aLut = aLut / (selectedRange[1]-selectedRange[0])
    
    imgTileRow = 0
    for k in range(numTiles[0] + 1):
        if k == 0:
            # special case: top row
            # always divisible by 2 because of padding
            imgTileNumRows = int(dimTile[0] / 2)
            mapTileRows = [0, 0]
        elif k == numTiles[0]:
            # special case: bottom row
            imgTileNumRows = int(dimTile[0] / 2)
            mapTileRows = [numTiles[0] - 1, numTiles[0] - 1]
        else:
            imgTileNumRows = int(dimTile[0])
            # [upperRow lowerRow]
            mapTileRows = [k - 1, k]
        
        # loop over columns of the tileMappings cell array
        imgTileCol = 0
        for l in range(numTiles[1] + 1):
            if l == 0:
                # special case: left column
                imgTileNumCols = int(dimTile[1] / 2)
                mapTileCols = [0, 0]
            elif l == numTiles[1]:
                # special case: right column
                imgTileNumCols = int(dimTile[1] / 2)
                mapTileCols = [numTiles[1] - 1, numTiles[1] - 1]
            else:
                imgTileNumCols = int(dimTile[1])
                # [right left]
                mapTileCols = [l - 1, l]
            
            # Extract four tile mappings
            ulMapTile = tileMappings[(mapTileRows[0], mapTileCols[0])]
            urMapTile = tileMappings[(mapTileRows[0], mapTileCols[1])]
            blMapTile = tileMappings[(mapTileRows[1], mapTileCols[0])]
            brMapTile = tileMappings[(mapTileRows[1], mapTileCols[1])]
            
            # Calculate the new greylevel assignments of pixels
            # within a submatrix of the image specified by imgTileIdx. This
            # is done by a bilinear interpolation between four different mappings
            # in order to eliminate boundary artifacts.
            normFactor = imgTileNumRows * imgTileNumCols # normalization factor
            
#             imgTileIdx = [range(imgTileRow,imgTileRow + imgTileNumRows), range(imgTileCol,imgTileCol + imgTileNumCols)]
            imgPixVals = grayxform(I[imgTileRow:imgTileRow + imgTileNumRows,
                                    imgTileCol:imgTileCol + imgTileNumCols], aLut)
            
            # calculate the weights used for linear interpolation between the
            # four mappings
            rowW = np.repeat(np.arange(0,imgTileNumRows).reshape((imgTileNumRows,1)),imgTileNumCols,axis=1)
            colW = np.repeat(np.arange(0,imgTileNumCols).reshape((1,imgTileNumCols)),imgTileNumRows,axis=0)
            rowRevW = np.repeat(np.arange(imgTileNumRows,0,-1).reshape((imgTileNumRows,1)),imgTileNumCols,axis=1)
            colRevW = np.repeat(np.arange(imgTileNumCols,0,-1).reshape((1,imgTileNumCols)),imgTileNumRows,axis=0)
            S1=grayxform(imgPixVals,ulMapTile).astype(float)
            S2=grayxform(imgPixVals,urMapTile).astype(float)
            S3=grayxform(imgPixVals,blMapTile).astype(float)
            S4=grayxform(imgPixVals,brMapTile).astype(float)
            
            claheI[imgTileRow:imgTileRow + imgTileNumRows, imgTileCol:imgTileCol + imgTileNumCols] = \
                (rowRevW * (colRevW * S1 + \
                            colW    * S2) + \
                 rowW    * (colRevW * S3 + \
                            colW    * S4)) / normFactor
            
            imgTileCol += imgTileNumCols
        imgTileRow += imgTileNumRows
    return claheI