clear
close all

% For reproducibility
rng(0)

%% Load scRNA-seq data with dropout
load Xdrop.mat

% PCA of raw data
[cff,X_pca_drop]=pca(zscore(Xdrop),'NumComponents',2);


FS=24;

% Plotting the PCs
figure
gscatter(X_pca_drop(:,1),X_pca_drop(:,2),groundTruth,[],[],[],'off')
ylabel('PC2')
xlabel('PC1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(Xdrop),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)

%% SERM parameters
Xpart=Xdrop(1:5000,:); % Part of the data for learning data distribution
maxEPOCH=20; % maximum epoch for autoencoder
ROIsize=[2000 1000]; % ROI size 
percOL=0.25; % Percent overlap between successive ROIs
reduced_Dim=2; % The size of the latent dimension in the autoencoder

% SERM operation
X_adapthisteq=serm_v3((Xdrop),Xpart,reduced_Dim,maxEPOCH,ROIsize,percOL);

% PCA of SERM-imputed data
[cff,X_pca_hydra]=pca(zscore(X_adapthisteq),'NumComponents',2);

figure
gscatter(X_pca_hydra(:,1),X_pca_hydra(:,2),groundTruth,[],[],[],'off')
ylabel('sPC2')
xlabel('sPC1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(X_adapthisteq),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)




