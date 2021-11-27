
function hgram=findDist3(X,reduced_Dim,maxEPOCH)



autoenc = trainAutoencoder(X','hiddenSize',reduced_Dim,'MaxEpochs',maxEPOCH,'DecoderTransferFunction','satlin',....
    'L2WeightRegularization',0.001,'ShowProgressWindow',false,'SparsityProportion',1,'SparsityRegularization',1.6,....
    'UseGPU',false);
finalOut_SCA = predict(autoenc,X');

Xrec = finalOut_SCA';

tempDist=0.01:0.01:1;
%hgram=betaa*exp(-betaa*tempDist);
NBin=length(tempDist);
Xn=rescale(Xrec);Xn(Xn>0.05)=0.05;
[N,edges,bin] = histcounts(rescale(Xn),NBin,'Normalization','probability');

gaussEqn = '(1/(sqrt(pi*n)*c))*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0.01],...
               'Upper',[0.1],...
               'StartPoint',[0.05]);
%[f2,gof2,outFit2] = fit(tempDist',N',expEqn,'options',fo);
ft = fittype(gaussEqn,'problem','n','options',fo);
[f1,gof1,outFit1] = fit(tempDist',N',ft,'problem',1);

expEqn = '(a^n)*exp(-(b*x))';
% fo = fitoptions('Lower',[0,5],...
%                'Upper',[Inf,20],...
%                'StartPoint',[1 1]);
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,5],...
               'Upper',[Inf,20],...
               'StartPoint',[1 5]);
%[f2,gof2,outFit2] = fit(tempDist',N',expEqn,'options',fo);
ft = fittype(expEqn,'problem','n','options',fo);
[f2,gof2,outFit2] = fit(tempDist',N',ft,'problem',1);

rayEqn =  '((2*x*n)/c^2)*exp(-(x/c)^2)';
fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0.01],...
               'Upper',[0.1],...
               'StartPoint',[0.05]);
%[f2,gof2,outFit2] = fit(tempDist',N',expEqn,'options',fo);
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

% 
 outF=outfVec{idx};
 
%  display('Learned distribution....')
%  f{idx}
 %f2
hgram=N'-outF.residuals;

% if idx==2
%     if f2.b<5
%         betaa=5;
%     elseif f2.b>20
%         betaa=20;
%     end
% hgram=betaa*exp(-betaa*tempDist);    
    
end