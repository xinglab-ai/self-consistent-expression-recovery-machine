%   findDistribution 
%   idealDistribution=findDistribution(X,reduced_Dim,maxEPOCH)
%   learns the distribution of the ideal data 
%
%   Inputs:
%   data: double matrix, gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the genes. 
%   reduced_Dim: latent dimension of the encoder-decoder network
%   maxEPOCH: epochs for the training of the network
%   Outputs:
%   idealDistribution: Distribution of the ideal expression data
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
function idealDistribution=findDistribution(data,reduced_Dim,maxEPOCH)

% Use the encoder-decoder network
autoenc = trainAutoencoder(data','hiddenSize',reduced_Dim,'MaxEpochs',maxEPOCH,'DecoderTransferFunction','satlin',....
    'L2WeightRegularization',0.001,'ShowProgressWindow',false,'SparsityProportion',1,'SparsityRegularization',1.6,....
    'UseGPU',false);
finalOut_SCA = predict(autoenc,data');

Xrec = finalOut_SCA';

tempDist=0.01:0.01:1;
NBin=length(tempDist);
Xn=rescale(Xrec);Xn(Xn>0.05)=0.05;
[N,edges,bin] = histcounts(rescale(Xn),NBin,'Normalization','probability');

gaussEqn = '(1/(sqrt(pi*n)*c))*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0.01],...
               'Upper',[0.1],...
               'StartPoint',[0.05]);

ft = fittype(gaussEqn,'problem','n','options',fo);
[f1,gof1,outFit1] = fit(tempDist',N',ft,'problem',1);

expEqn = '(a^n)*exp(-(b*x))';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,5],...
               'Upper',[Inf,20],...
               'StartPoint',[1 5]);
ft = fittype(expEqn,'problem','n','options',fo);
[f2,gof2,outFit2] = fit(tempDist',N',ft,'problem',1);

rayEqn =  '((2*x*n)/c^2)*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0.01],...
               'Upper',[0.1],...
               'StartPoint',[0.05]);

ft = fittype(rayEqn,'problem','n','options',fo);
[f3,gof3,outFit3] = fit(tempDist',N',ft,'problem',1);

gofs=[gof1.rmse gof2.rmse gof3.rmse];
outfVec{1}=outFit1;
outfVec{2}=outFit2;
outfVec{3}=outFit3;

f{1}=f1;
f{2}=f2;
f{3}=f3;

idx=find(gofs==min(gofs));
 outF=outfVec{idx}; 
idealDistribution=N'-outF.residuals;
end