% extract_fv(3,'spatial')
function  extract_fv(scale,tag)
addpath(genpath('/home/civic.org.cn/zyz/md128/toolbox'));
%input:
%   layer:3 or 5

% ############################################### %
% configure
data_dir = '/home/civic.org.cn/zyz/md128/HMDB51';
log_file = ['/home/civic.org.cn/zyz/md128/HMDB51/',tag,'fv.log'];
% ############################################### %


path_tdd = fullfile(data_dir,[tag,'_spatial_scale_',scale]);
fv_dir = fullfile(data_dir,['fv_', tag]);

dim = 64;
num = 256;
pca_sample = 100;

if ~exist(path_tdd,'dir')          % check dir validation
    error(['tdd dir:',path_tdd,' not exist!']); 
end

if ~exist(fv_dir,'dir')
    mkdir(fv_dir);
end

fid = fopen(log_file,'w');
fprintf(fid,'%s\n',datestr(now,0));
log_exist = ['exist file:',char(13,10)'];
log_error = ['error file:',char(13,10)'];

function folderlist = getFolderList(ddir)
    folderlist = dir(ddir);
    folderlist = {folderlist(:).name};
    folderlist = setdiff(folderlist,{'.','..'});
    folderlist = folderlist;
end

function [U,mu,means, covariances, priors] = extract_pca(tdd_dir,d,numCluster,sample_num)
    
    classes = getFolderList(tdd_dir);
	pcatrain = {};
	gmmtrain = {};

	for cclassname=classes
% 		display(classname);
        classname=cclassname{:};
        tddfiles = getFolderList(fullfile(tdd_dir,classname,'*.mat'));
        for cfilename = tddfiles
            filename =cfilename{:};
			tdd_feature = load(fullfile(tdd_dir,classname,filename));
            tdd_feature = tdd_feature.idt_cnn_feature;
            %
            for kk = linspace(1,4,4)
                pcatrain{kk} = [pcatrain{kk} datasample(tdd_feature{kk},sample_num,2)]; 
    %             size(pcatrain)
                gmmtrain{kk} = [gmmtrain{kk} datasample(tdd_feature{kk},sample_num,2)];  
            end
        end
	end
% 	save '/data1/fisher/pcatrain.mat' pcatrain;
% 	save '/data1/fisher/gmmtrain.mat' gmmtrain;

    for kk = linspace(1,4,4)
        display('PCA Started');
        [U{kk},mu{kk},~] = pca(pcatrain{kk});   % calculate parameter neended for pca
        display('PCA Complete');
        gmmtrain_pca = pcaApply(gmmtrain{kk},U{kk},mu{kk},d);  % get pca code 
        display('GMM Trainset Ready');
        [means{kk}, covariances{kk}, priors{kk}] = vl_gmm(gmmtrain_pca, numCluster) ;  % get gmm parameter
        display('GMM Trained');
    end
	
% 	save '/data1/fisher/pca_space.mat' U mu; 
	
% 	save '/data1/fisher/gmm_features.mat' means covariances priors; 	
end

[U,mu,means, covariances, priors] = extract_pca(path_tdd,dim,num,pca_sample);

tic;

folderlist = getFolderList(path_tdd);
display('fv started ...');

for cfoldername=folderlist
	display(['proceeding', cfoldername]);
    foldername=cfoldername{:};
    if ~exist(fullfile(fv_dir,foldername),'dir')
        mkdir(fullfile(fv_dir,foldername));
    end
    
    filelist = getFolderList(fullfile(path_tdd,foldername,'*.mat'));
    for ctddfile = filelist
        tddfile = ctddfile{:};
%         if strcmp(tddfile,'v_shooting_10_03.mat')
%             display('stop')
%         end
        
        fv_file = fullfile(fv_dir,foldername,[tddfile(1:end-4),'.mat'])
        if exist(fv_file)
            log_exist = [log_exist,fv_file,char(13,10)'];
            continue;
        end
        
        try    
            tddfeature = load(fullfile(path_tdd{i},foldername,tddfile));
            tddfeature = tddfeature.idt_cnn_feature;
            for i=linspace(1,4,4)    
                tddfeature_pca = pcaApply(tddfeature{i},U{i},mu{i},dim); 
                tddfeature_gmm{i} = vl_fisher(single(tddfeature_pca),single(means{i}),single(covariances{i}),single(priors{i}));
    % 			gmmtrain = [gmmtrain datasample(tdd_feature_spatial_conv4_norm_1,6,2) datasample(tdd_feature_spatial_conv4_norm_2,6,2) datasample(tdd_feature_spatial_conv5_norm_1,6,2) datasample(tdd_feature_spatial_conv5_norm_2,6,2) ]; 
            end
            save(fv_file,'tddfeature_gmm');
        catch
            log_error = [log_error,tddfile,char(13,10)'];
        end
    end     
end
toc;
fprintf(fid,'%s\n%s',log_exist,log_error);
fclose(fid);
end

