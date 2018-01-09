function  extract_fv()
addpath(genpath('/home/chang/toolbox'));
%input:
%   layer:3 or 5

path_tdd{1} = '/data/UCF11/tdd_spatial_scale_3_conv4';
path_tdd{2} = '/data/UCF11/tdd_spatial_scale_3_conv5';
fv_dir = '/data/UCF11/fv_spatial';

dim = 64
num = 256;
if exist(fv_dir,'dir')
    error('fv dir exist!')
else
    mkdir(fv_dir);
end

function folderlist = getFolderList(ddir)
    folderlist = dir(ddir);
    folderlist = {folderlist(:).name};
    folderlist = setdiff(folderlist,{'.','..'});
end

function [U,mu,means, covariances, priors] = extract_pca(tdd_dir,d,numCluster)
    
    classes = getFolderList(fullfile(tdd_dir));
	pcatrain = [];
% 	gmmtrain = [];

	for classname=classes
% 		display(classname);
        tddfiles = getFolderList([fullfile(tdd_dir,classname,'*.mat')]);
        for filename = tddfiles
			tdd_feature = load(fullfile(tdd_dir,classname,tddfiles));
            tdd_feature = tdd_feature.idt_cnn_feature;
			pcatrain = [pcatrain tdd_feature]; 
% 			gmmtrain = [gmmtrain datasample(tdd_feature_spatial_conv4_norm_1,6,2) datasample(tdd_feature_spatial_conv4_norm_2,6,2) datasample(tdd_feature_spatial_conv5_norm_1,6,2) datasample(tdd_feature_spatial_conv5_norm_2,6,2) ]; 
        end
	end
% 	save '/data1/fisher/pcatrain.mat' pcatrain;
% 	save '/data1/fisher/gmmtrain.mat' gmmtrain;
	display('PCA Started');
	[U,mu,~] = pca(pcatrain);   % calculate parameter neended for pca
	display('PCA Complete');
% 	save '/data1/fisher/pca_space.mat' U mu; 
	gmmtrain_pca = pcaApply(pcatrain,U,mu,d);  % get pca code 
	display('GMM Trainset Ready');
	[means, covariances, priors] = vl_gmm(gmmtrain_pca, numCluster) ;  % get gmm parameter
	display('GMM Trained');
% 	save '/data1/fisher/gmm_features.mat' means covariances priors; 	
end

[U{1},mu{1},means{1}, covariances{1}, priors{1}] = extract_pca(path_tdd{1},dim,num);
[U{2},mu{2},means{2}, covariances{2}, priors{2}] = extract_pca(path_tdd{2},dim,num);

tic;

folderlist = getFolderList(path_tdd{1});
for foldername=folderlist
% 		display(classname);
    filelist = getFolderList(fullfile(path_tdd{1},foldername,'*.mat'));
    for tddfile = filelist
        for i=[1,2]
            if ~exist([path_tdd{i},foldername],'dir')
                mkdir([path_tdd{i},foldername]);
            end
            tddfeature_gmm = zeros(1,2*num*dim);
            tddfeature = load(fullfile(path_tdd{i},foldername,tddfile));
            tddfeature = tddfeature.idt_cnn_feature;
            tddfeature_pca = pcaApply(tddfeature,U{i},mu{i},64); 
            tddfeature_gmm{i} = vl_fisher(tddfeature_pca,means{i},covariances{i},priors{i});
% 			gmmtrain = [gmmtrain datasample(tdd_feature_spatial_conv4_norm_1,6,2) datasample(tdd_feature_spatial_conv4_norm_2,6,2) datasample(tdd_feature_spatial_conv5_norm_1,6,2) datasample(tdd_feature_spatial_conv5_norm_2,6,2) ]; 
        end
        save(fullfile(fv_dir,foldername,[filelist(1:end-4),'.mat']),'tddfeature_gmm');
    end     
end
toc;
end

