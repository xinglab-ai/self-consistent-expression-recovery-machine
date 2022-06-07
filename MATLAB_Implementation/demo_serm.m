clear
close all
% For reproducibility
rng(0)

%% Load downsampled scRNA-seq data 
load data_CellTax_Downsampled_1perc.mat

%% SERM parameters

ROIsize=[12000 600]; % ROI size 
percOL=0.5; % Percent overlap between successive ROIs
Randomize=0; % Randomize set to True
% SERM operation
sermOut=serm(dataSample,ROIsize,percOL,Randomize);

% Load the original data for assessing the goodness of recovery
load data_CellTax_Original

sz=size(dataRef);
for i=1:sz(2)    
    acorr=corrcoef(dataSample(:,i),dataRef(:,i));    
    CorrVecORG(i)=acorr(1,2);
    bcorr=corrcoef(sermOut(:,i),dataRef(:,i));    
    CorrVecSERM(i)=bcorr(1,2);  
end

%% Plotting Pearson correlation values

Y=[CorrVecORG' CorrVecSERM'];
cats={'Downsampled','SERM'};
fcolor=[0 0 1;1 0 0];

figure
vs = violinplotT(Y, cats,fcolor);
ylabel('Pearson coefficient')
set(gca,'FontSize',20)



