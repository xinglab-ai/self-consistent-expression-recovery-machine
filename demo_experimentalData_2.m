clear
close all

% For reproducibility
rng(0)

%% Load scRNA-seq data with dropout
load experimentalData_3_StemCell.mat

% t-SNE of raw data
[Xtsne_RAW]=tsne(zscore(data));


FS=24;

% Plotting the PCs
figure
gscatter(Xtsne_RAW(:,1),Xtsne_RAW(:,2),groundTruth,[],[],[],'off')
ylabel('tSNE2')
xlabel('tSNE1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(data),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)

%% SERM parameters
Xpart=data(1:2000,:); % Part of the data for learning data distribution
maxEPOCH=20; % maximum epoch for autoencoder
ROIsize=[2000 500]; % ROI size 
percOL=0.5; % Percent overlap between successive ROIs
reduced_Dim=2; % The size of the latent dimension in the autoencoder

% SERM operation
sermOut=serm(data,Xpart,reduced_Dim,maxEPOCH,ROIsize,percOL);

% use sermOut=serm_v3((data),Xpart,reduced_Dim,maxEPOCH,ROIsize,percOL)
% for better speed


% t-SNE of SERM-imputed data
[Xtsne_SERM]=tsne(zscore(sermOut));

figure
gscatter(Xtsne_SERM(:,1),Xtsne_SERM(:,2),groundTruth,[],[],[],'off')
ylabel('stSNE2')
xlabel('stSNE1')
set(gca,'FontSize',FS)

% Histogram computation & plotting
NBin=30;
[N,edges,bin] = histcounts(rescale(sermOut),NBin,'Normalization','probability');

binX=(1:NBin)/NBin;

figure
bar(binX,N,'histc')
ylabel('Normalized frequency')
xlabel('Normalized value')
set(gca,'FontSize',FS)




