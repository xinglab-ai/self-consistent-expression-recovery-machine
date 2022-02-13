#   SERM 
#
#   Produces a recovered gene expression matrix from expressions with dropout.
#
#   Written by Md Tauhidul Islam and Hongyi Ren, Radiation Oncology,
#   Stanford University

import os
from functools import partial

import numpy as np
import scipy
from scipy.optimize import curve_fit
from sparseae import *

def serm(data, dataPartLearning, reduced_Dim, maxEPOCH, ROIsize, percOL):
    # produces a recovered gene expression matrix from expressions with dropout.
    # 
    # Inputs:
    #  data: double matrix, gene expression data 
    #    in tabular format, i.e., rows denote cells and columns denote the genes. 
    #  dataPartLearning: idouble matrix, gene expression data 
    #    Part of the expression matrix used for learning the data distribution 
    #  reduced_Dim: latent dimension of the encoder-decoder network
    #  maxEPOCH: epochs for the training of the network
    #  ROIsize: Height and width of the ROIs used in SERM to recover the
    #    expression values
    #  percOL: Percent overlap between two consecutive ROIs. The value should
    #    between 0 and 1. 
    # 
    # Return:
    #    recoveredMatrix: SERM-recovered gene expression matrix
    
    SCA_X = trainAutoencoder(dataPartLearning, reduced_Dim, maxEPOCH)
    idealDistribution = findDistribution(SCA_X)
    sermOut = recovery(data, idealDistribution, ROIsize, percOL)
    
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
    
def gauss_func(x, c):
    # Gaussian Distribution
    # (1/(sqrt(pi)*c))*exp(-(x/c)^2)
    return (1/(np.sqrt(np.pi*c)))*np.exp(-(x/c)**2)

def exp_func(x, a, b):
    # Exponential Distribution
    # a*exp(-(b*x))
    return a*np.exp(-(b*x))

def ray_func(x, c):
    # Rayleigh distribution
    # ((2*x)/c^2)*exp(-(x/c)^2)
    return ((2*x)/(c**2))*np.exp(-(x/c)**2)

def _match_hist(img, hist):
    # histogram matching algorithm which is same as histeq(img, hist) in matlab,
    # which will transform the source image to have a close histogram as given 
    # histogram.
    #
    # Input:
    #  img: source image
    #  hist: target hitogram
    # 
    # Return:
    #  images that having similar histogram as hist
    n = 256
    m = hist.size
    
    img_hist, bin_edges = np.histogram(img, bins=n,range=(0,1))
    img_cum = np.cumsum(img_hist)

    hist_cum = np.cumsum(hist) * img.size / np.sum(hist)
    
    tol = np.ones((m,1)) * \
          np.min([np.append(img_hist[0:-1],[0]),
                  np.append([0],img_hist[1:])],axis=0)/2

    err = (hist_cum.reshape((m,1))*np.ones((1,n))-\
           np.ones((m,1))*img_cum.reshape((1,n)))+tol
    err[err < -img.size * np.sqrt(np.finfo(float).eps)] = img.size
    
    T = np.argmin(err,axis=0)
    T = T/(m-1)
    maxTidx = T.size-1

    return T[(img * maxTidx + 0.5).astype(int)]

def findDistribution(X):
    # learns the distribution of the ideal data 
    #
    # Inputs:
    #  X: the data that is transformed by a Sparse Autoencoder,
    #   each row is a data sample
    # 
    # Return:
    #  idealDistribution: Distribution of the ideal expression data
    Xrec = X.T
    tempDist = np.arange(0.01,1.01,0.01)

    NBin = tempDist.shape[0]

    Xn = (Xrec-np.min(Xrec)) / (np.max(Xrec)-np.min(Xrec))
    Xn[Xn>0.05] = 0.05;
    Xn_ = (Xn-np.min(Xn)) / (np.max(Xn)-np.min(Xn))
    [N,edges] = np.histogram(Xn_,bins=NBin,density=True)
    probs = N * 0.01

    gaussCoeff,_ = curve_fit(gauss_func, tempDist, probs,
                             bounds=([0.01],[0.1]), p0=0.05)
    gaussEqn = partial(gauss_func, c=gaussCoeff)

    min_rmse = computeRMSE(tempDist, probs, gaussEqn)
    residuals = computeResiduals(tempDist, probs, gaussEqn)

    expCoeff,_ = curve_fit(exp_func, tempDist, probs,
                           bounds=([0., 5.],[np.inf, 20.]), p0=[1,5])
    expEqn = partial(exp_func, a=expCoeff[0], b=expCoeff[1])

    if computeRMSE(tempDist, probs, expEqn) < min_rmse:
        min_rmse = computeRMSE(tempDist, probs, expEqn)
        residuals = computeResiduals(tempDist, probs, expEqn)

    rayCoeff,_ = curve_fit(ray_func, tempDist, probs,
                           bounds=([0.01],[0.1]), p0=0.05)
    rayEqn = partial(ray_func, c=rayCoeff)

    if computeRMSE(tempDist, probs, rayEqn) < min_rmse:
        min_rmse = computeRMSE(tempDist, probs, rayEqn)
        residuals = computeResiduals(tempDist, probs, rayEqn)
        
    return (probs - residuals).reshape([100,1])


def recovery(data, idealDistribution, ROIsize, percOL):
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
            Xh = _match_hist(Xtake,idealDistribution)
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