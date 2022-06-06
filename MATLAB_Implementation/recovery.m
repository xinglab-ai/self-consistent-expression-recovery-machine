%   recovery 
%   recoveredMatrix=recovery(data,idealDistribution,ROIsize,percOL)
%    produces a recovered gene expression matrix from expressions with dropout
%    and ideal data distribution.
%
%   Inputs:
%   data: double matrix, gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the genes. 
%   idealDistribution: double vector 
%   Ideal distribution of the data without dropout
%    
%   ROIsize: Height and width of the ROIs used in SERM to recover the
%   expression values
%   percOL: Percent overlap between two consecutive ROIs. The value should
%   between 0 and 1. 
%   Outputs:
%   recoveredMatrix: SERM-recovered gene expression matrix
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu


function recoveredMatrix=recovery(data,distName,param,ROIsize,percOL)


[rs,cs]=size(data);
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


Xpad = padarray(data,[(rsEdit-rs)/2 (csEdit-cs)/2],0,'both');

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


M=50;
meanX=mean(X(:));
X(X>M*meanX)=M*meanX;


minX=min(X(:));
maxX=max(X(:));

Xre=rescale(X);


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
        Xh = adapthisteqSERM(Xtake,'NumTiles',[2 2],'ClipLimit',0.0,'NBins',256,'Distribution',distName,'Alpha',param);
        XpadSave((i-1)*overlappixelX+1:(i-1)*overlappixelX+ROIsize(1),(j-1)*overlappixelY+1:(j-1)*overlappixelY+ROIsize(2))=Xh;
    end
end

        
 Xhist=XpadSave(round((numItX2*overlappixelX-sx)/2)+1:end-round((numItX2*overlappixelX-sx)/2),round((numItY2*overlappixelY-sy)/2)+1:end-round((numItY2*overlappixelY-sy)/2));
Xk=Xhist-min(Xhist(:));
Y=minX+(maxX-minX)*Xk;
Ypad2 = Y((rsEdit2-rsEdit)/2+1:end-(rsEdit2-rsEdit)/2,(csEdit2-csEdit)/2+1:end-(csEdit2-csEdit)/2);
recoveredMatrix = Ypad2((rsEdit-rs)/2+1:end-(rsEdit-rs)/2,(csEdit-cs)/2+1:end-(csEdit-cs)/2);


end
