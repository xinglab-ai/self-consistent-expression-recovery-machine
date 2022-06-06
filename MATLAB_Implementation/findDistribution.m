%   findDistribution 
%   idealDistribution=findDistribution(X,reduced_Dim,maxEPOCH)
%   learns the distribution of the ideal data 
%
%   Inputs:
%   dataPart: double matrix, gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the genes. 
%   reduced_Dim: latent dimension of the encoder-decoder network
%   maxEPOCH: epochs for the training of the network
%   Outputs:
%   distName: Distribution of the denoised expression data
%   param: parameter of the distribution
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
function [distName,param]=findDistribution(dataPart,reduced_Dim,maxEPOCH)
% Use the encoder-decoder network


autoenc = trainAutoencoder(dataPart','hiddenSize',reduced_Dim,'MaxEpochs',maxEPOCH,'DecoderTransferFunction','satlin',....
    'L2WeightRegularization',0.05,'ShowProgressWindow',false,'SparsityProportion',0.9,'SparsityRegularization',1,....
    'UseGPU',false);
finalOut_SCA = predict(autoenc,dataPart');
Xrec = finalOut_SCA';
tempDist=0.01:0.01:1;
NBin=length(tempDist);
[N,edges,bin] = histcounts(rescale(Xrec),NBin,'Normalization','probability');
tempDist=tempDist(2:end);
N=N(2:end);

gaussEqn = '(P/(sqrt(pi*n)*c))*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,0.1],...
               'Upper',[Inf,20],...
               'StartPoint',[1,5]);
ft = fittype(gaussEqn,'problem','n','options',fo);
[f1,gof1,outFit1] = fit(tempDist',N',ft,'problem',1);

expEqn = '(a^n)*exp(-(b*x))';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,5],...
               'Upper',[Inf,20],...
               'StartPoint',[1 5]);
ft = fittype(expEqn,'problem','n','options',fo);
[f2,gof2,outFit2] = fit(tempDist',N',ft,'problem',1);

rayEqn =  '((P*2*x*n)/c^2)*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,0.01],...
               'Upper',[Inf,0.5],...
               'StartPoint',[1,0.2]);
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
 
if idx==1
    hgram=(1/(sqrt(pi)*(f1.c))).*exp(-(tempDist/(f1.c)).^2);
    distName='gaussian';
    param=f1.c/sqrt(2); % To correct for the scaling in the fitting (c=sigma*sqrt(2))
elseif idx==2
    hgram=(f2.b)*exp(-(f2.b*tempDist));
    distName='exponential';
    param=f2.b;
else
    hgram=((2*tempDist)/(f3.c)^2).*exp(-(tempDist/(f3.c)).^2);
    distName='rayleigh';
    param=f3.c/sqrt(2); % To correct for the scaling in the fitting (c=sigma*sqrt(2))
end

display('Learned distribution:')
distName

display('Optimized parameter:')
param
end


