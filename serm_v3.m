
function Y=serm_v3(Xorg,Xpart,reduced_Dim,maxEPOCH,ROIsize,percOL)

hgram=findDist3(Xpart,reduced_Dim,maxEPOCH);

Y=chydra3(Xorg,hgram,ROIsize,percOL);

end



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







function Ypad=chydra3(Xorg,hgram,ROIsize,percOL)


[rs,cs]=size(Xorg);
numROIr=ceil(rs/ROIsize(1));
numROIc=ceil(cs/ROIsize(2));

rsEdit=numROIr*ROIsize(1);
csEdit=numROIc*ROIsize(2);

if (rem((rsEdit-rs),2))
    rsEdit=rsEdit+1;
end

if (rem((csEdit-cs),2))
    csEdit=csEdit+1;
end


Xpad = padarray(Xorg,[(rsEdit-rs)/2 (csEdit-cs)/2],0,'both');

[rsp,csp]=size(Xpad);
OLsize=ceil(ROIsize*percOL);

numROIrp=ceil(rsp/OLsize(1));
numROIcp=ceil(csp/OLsize(2));


rsEdit2=numROIrp*OLsize(1);
csEdit2=numROIcp*OLsize(2);

if (rem((rsEdit2-rsEdit),2))
    rsEdit2=rsEdit2+1;
end

if (rem((csEdit2-csEdit),2))
    csEdit2=csEdit2+1;
end

Xpad2 = padarray(Xpad,[(rsEdit2-rsEdit)/2 (csEdit2-csEdit)/2],0,'both');

X=Xpad2;

% tempDist=0.01:0.01:1;
% hgram=betaa*exp(-betaa*tempDist);

M=50;
meanX=mean(X(:));
X(X>M*meanX)=M*meanX;


minX=min(X(:));
maxX=max(X(:));

Xre=rescale(X);

%X_adapthisteq = adapthisteq(Xre,'NumTiles',[2 2],'ClipLimit',0.2,'NBins',256,'Distribution','exponential','Alpha',5);

[sx,sy]=size(X);
overlappixelX=ceil(percOL*ROIsize(1));
overlappixelY=ceil(percOL*ROIsize(2));
numItX2=ceil(sx/overlappixelX);
numItY2=ceil(sy/overlappixelY);


numItX=ceil(sx/overlappixelX)-(1/percOL)+1;
numItY=ceil(sy/overlappixelY)-(1/percOL)+1;



Xpad = padarray(Xre,[round((numItX2*overlappixelX-sx)/2) round((numItY2*overlappixelY-sy)/2)],0,'both');

XpadSave=Xpad;

for i=1:numItX
    for j=1:numItY
        Xtake=Xpad((i-1)*overlappixelX+1:(i-1)*overlappixelX+ROIsize(1),(j-1)*overlappixelY+1:(j-1)*overlappixelY+ROIsize(2));
        Xh = histeq(Xtake,hgram);
        XpadSave((i-1)*overlappixelX+1:(i-1)*overlappixelX+ROIsize(1),(j-1)*overlappixelY+1:(j-1)*overlappixelY+ROIsize(2))=Xh;
    end
end

        
 Xhist=XpadSave(round((numItX2*overlappixelX-sx)/2)+1:end-round((numItX2*overlappixelX-sx)/2),round((numItY2*overlappixelY-sy)/2)+1:end-round((numItY2*overlappixelY-sy)/2));


Xk=Xhist-min(Xhist(:));

Y=minX+(maxX-minX)*Xk;


Ypad2 = Y((rsEdit2-rsEdit)/2+1:end-(rsEdit2-rsEdit)/2,(csEdit2-csEdit)/2+1:end-(csEdit2-csEdit)/2);

Ypad = Ypad2((rsEdit-rs)/2+1:end-(rsEdit-rs)/2,(csEdit-cs)/2+1:end-(csEdit-cs)/2);


end
