%   SERM 
%    recoveredMatrix=serm(data,dataPartLearning,reduced_Dim,maxEPOCH,ROIsize,percOL)
%    produces a recovered gene expression matrix from expressions with dropout.
%
%   Inputs:
%   data: double matrix, gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the genes. 
%   dataPartLearning: idouble matrix, gene expression data 
%   Part of the expression matrix used for learning the data distribution 
%    
%   reduced_Dim: latent dimension of the encoder-decoder network
%   maxEPOCH: epochs for the training of the network
%   ROIsize: Height and width of the ROIs used in SERM to recover the
%   expression values
%   percOL: Percent overlap between two consecutive ROIs. The value should
%   between 0 and 1. 
%   Outputs:
%   recoveredMatrix: SERM-recovered gene expression matrix
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
%%
function recoveredMatrix=serm(data,dataPartLearning,reduced_Dim,maxEPOCH,ROIsize,percOL)

% Learn the distribution of the ideal data
idealDistribution=findDistribution(dataPartLearning,reduced_Dim,maxEPOCH);

% Recover the expression values by matching the distribution of dropout-
% affected values to the ideal data values
recoveredMatrix=recovery(data,idealDistribution,ROIsize,percOL);

end







