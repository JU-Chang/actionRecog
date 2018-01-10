function  extract_fv()
addpath(genpath('/home/chang/toolbox'));
%input:
%   layer:3 or 5

path_tdd{1} = '/data/UCF11/tdd_spatial_scale_3_conv4';
path_tdd{2} = '/data/UCF11/tdd_spatial_scale_3_conv5';
fv_dir = '/data/UCF11/fv_spatial';

dim = 64;
num = 256;
pca_sample = 100;
if exist(fv_dir,'dir')
    error('fv dir exist!')
end

function folderlist = getFolderList(ddir)
    folderlist = dir(ddir);
    folderlist = {folderlist(:).name};
    folderlist = setdiff(folderlist,{'.','..'});
    folderlist = folderlist;
end

function [U,mu,means, covariances, priors] = extract_pca(tdd_dir,d,numCluster,sample_num)
    
    classes = getFolderList(tdd_dir);
	pcatrain = [];
	gmmtrain = [];

	for cclassname=classes
% 		display(classname);
        classname=cclassname{:};
        tddfiles = getFolderList(fullfile(tdd_dir,classname,'*.mat'));
        for cfilename = tddfiles
            filename =cfilename{:};
			tdd_feature = load(fullfile(tdd_dir,classname,filename));
            tdd_feature = tdd_feature.idt_cnn_feature;
            %
			pcatrain = [pcatrain datasample(tdd_feature,sample_num,2)]; 
%             size(pcatrain)
            gmmtrain = [gmmtrain datasample(tdd_feature,sample_num,2)];         
        end
	end
% 	save '/data1/fisher/pcatrain.mat' pcatrain;
% 	save '/data1/fisher/gmmtrain.mat' gmmtrain;
	display('PCA Started');
	[U,mu,~] = pca(pcatrain);   % calculate parameter neended for pca
	display('PCA Complete');
% 	save '/data1/fisher/pca_space.mat' U mu; 
	gmmtrain_pca = pcaApply(gmmtrain,U,mu,d);  % get pca code 
	display('GMM Trainset Ready');
	[means, covariances, priors] = vl_gmm(gmmtrain_pca, numCluster) ;  % get gmm parameter
	display('GMM Trained');
% 	save '/data1/fisher/gmm_features.mat' means covariances priors; 	
end

display('pca started ...');
[U{1},mu{1},means{1}, covariances{1}, priors{1}] = extract_pca(path_tdd{1},dim,num,pca_sample);
[U{2},mu{2},means{2}, covariances{2}, priors{2}] = extract_pca(path_tdd{2},dim,num,pca_sample);
display('pca complete!');

tic;

mkdir(fv_dir);
folderlist = getFolderList(path_tdd{1});
display('fv started ...');
for cfoldername=folderlist
	display(['proceeding', cfoldername]);
    foldername=cfoldername{:};
    if ~exist(fullfile(fv_dir,foldername),'dir')
        mkdir(fullfile(fv_dir,foldername));
    end
    filelist = getFolderList(fullfile(path_tdd{1},foldername,'*.mat'));
    for ctddfile = filelist
        tddfile = ctddfile{:};
        if strcmp(tddfile,'v_shooting_10_03.mat')
            display('stop')
        end
        for i=[1,2]    
            tddfeature = load(fullfile(path_tdd{i},foldername,tddfile));
            tddfeature = tddfeature.idt_cnn_feature;
            tddfeature_pca = pcaApply(tddfeature,U{1,i},mu{1,i},64); 
            tddfeature_gmm{i} = vl_fisher(single(tddfeature_pca),single(means{1,i}),single(covariances{1,i}),single(priors{1,i}));
% 			gmmtrain = [gmmtrain datasample(tdd_feature_spatial_conv4_norm_1,6,2) datasample(tdd_feature_spatial_conv4_norm_2,6,2) datasample(tdd_feature_spatial_conv5_norm_1,6,2) datasample(tdd_feature_spatial_conv5_norm_2,6,2) ]; 
        end
        save(fullfile(fv_dir,foldername,[tddfile(1:end-4),'.mat']),'tddfeature_gmm');
    end     
end
toc;
end

