%   SERM
%    recoveredMatrix=serm(data,dataPartLearning,reduced_Dim,maxEPOCH,ROIsize,percOL)
%    produces a recovered gene expression matrix from expressions with dropout.
%
%   Inputs:
%   data: double matrix, gene expression data
%   in tabular format, i.e., rows denote cells and columns denote the genes.
%   ROIsize: Height and width of the ROIs used in SERM to recover the
%   expression values
%   percOL: Percent overlap between two consecutive ROIs. The value should
%   between 0 and 1.
%   Randomize: if equal to 1(True), the data will be randomized both in 
%   column and row directions. 
%   Outputs:
%   recoveredMatrix: SERM-recovered gene expression matrix
%
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
%%
function recoveredMatrix=serm(data,ROIsize,percOL,Randomize)

sz=size(data);

if (Randomize==1)
    p = reshape(randperm(numel(data)),sz);
    dataRand = data(p);
    [~,inxRem] = sort(p(:));
else
    dataRand=data;
end

% Select 2000 cell data randomly for learning the distribution of denoised
% data
idxRand=randperm(sz(1),2000);
dataPartLearning=data(idxRand,:);
% Learn the distribution of the denoised data
%   reduced_Dim: latent dimension of the encoder-decoder network
%   maxEPOCH: epochs for the training of the network
reduced_Dim=2;
maxEPOCH=20;
[distName,param]=findDistribution(dataPartLearning,reduced_Dim,maxEPOCH);

% Recover the expression values by matching the distribution of dropout-
% affected values to the denoised data values
if (percOL==0)
    X=dataRand;
    minX=min(X(:));
    maxX=max(X(:));
    Xre=rescale(X);
    
    numTiles=ceil(sz./ROIsize);
    X_adapthisteq = adapthisteqSERM(Xre,'NumTiles',numTiles,'ClipLimit',0.0,'NBins',256,'Distribution',distName,'Alpha',param);
    Xk=X_adapthisteq-min(X_adapthisteq(:));
    recoveredRandMatrix=minX+(maxX-minX)*Xk;
else
    recoveredRandMatrix=recovery(dataRand,distName,param,ROIsize,percOL);
end

if (Randomize==1)
    recoveredMatrix= recoveredRandMatrix(reshape(inxRem,sz));
else
    recoveredMatrix=recoveredRandMatrix;
end
% Scaling to match the original data
recoveredMatrix=recoveredMatrix*(sum(data(:))/sum(recoveredMatrix(:)));
end







