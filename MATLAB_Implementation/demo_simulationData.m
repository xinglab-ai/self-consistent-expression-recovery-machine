clear
close all

% For reproducibility
rng(0)

%% Load Splatter simulated scRNA-seq data with dropout
load splatterSimulatedData.mat

% PCA of raw data
[cff,X_pca_drop]=pca(zscore(data_Dropout),'NumComponents',2);


FS=24;

% Plotting the PCs
figure
gscatter(X_pca_drop(:,1),X_pca_drop(:,2),groundTruth,[],[],[],'off')
ylabel('PC2')
xlabel('PC1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(data_Dropout),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)

%% SERM parameters
Xpart_Learning=data_Dropout(1:5000,:); % Part of the data for learning data distribution.
% The user can use any part of the data for learning. 
maxEPOCH=20; % maximum epoch for encoder-decoder network
ROIsize=[2000 1000]; % ROI size 
percOL=0.25; % Percent overlap between successive ROIs
reduced_Dim=2; % The size of the latent dimension in the encoder-decoder network

% SERM operation
serm_Out=serm_v3(data_Dropout,Xpart_Learning,reduced_Dim,maxEPOCH,ROIsize,percOL);
% The user can also use serm.m. serm_v3 is in .p format and has better
% speed than serm.m. However, they are same codes. 

% PCA of SERM-imputed data
[cff,X_pca_hydra]=pca(zscore(serm_Out),'NumComponents',2);

figure
gscatter(X_pca_hydra(:,1),X_pca_hydra(:,2),groundTruth,[],[],[],'off')
ylabel('sPC2')
xlabel('sPC1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(serm_Out),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)




