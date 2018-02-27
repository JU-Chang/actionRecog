% extract_fv(3,'spatial')
function  extract_fv(scale,tag)
addpath(genpath('/home/civic.org.cn/zyz/md128/toolbox'));
run('/home/civic.org.cn/zyz/vlfeat/toolbox/vl_setup')
%input:
%   layer:3 or 5

% ############################################### %
% configure
data_dir = '/home/civic.org.cn/zyz/md128/HMDB51';
log_file = ['/home/civic.org.cn/zyz/md128/HMDB51/',tag,'fv.log'];
% ############################################### %

dim = 64;
num = 256;
pca_sample = 6;
fv_dir = fullfile(data_dir,['pnorm_fv_', tag,'_psam_',num2str(pca_sample),'_dim_',num2str(dim)]);
pca_gmm = fullfile(data_dir,['temp/pca_gmm_psam_',num2str(pca_sample),'_dim_',num2str(dim),'.mat']);
path_tdd = fullfile(data_dir,['tdd_',tag,'_scale_',num2str(scale)]);


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
	pcatrain = {[],[],[],[]};
	gmmtrain = {[],[],[],[]};
    
    if exist(['temp/pcatrain','_p_',num2str(pca_sample),'_d_',num2str(dim),'.mat'])
        tmp1 = load(['temp/pcatrain','_p_',num2str(pca_sample),'_d_',num2str(dim),'.mat']);
        pcatrain = tmp1.pcatrain;
        tmp2 = load(['temp/gmmtrain','_p_',num2str(pca_sample),'_d_',num2str(dim),'.mat']);
        gmmtrain = tmp2.gmmtrain;
    else
        for cclassname=classes
    % 		display(classname);
            classname=cclassname{:};
            tddfiles = getFolderList(fullfile(tdd_dir,classname,'*.mat'));
            for cfilename = tddfiles
                filename =cfilename{:};
                tdd_feature = load(fullfile(tdd_dir,classname,filename));
                tdd_feature = tdd_feature.idt_cnn_feature;

                try
                    for kk = linspace(1,4,4)
                        pcatrain{kk} = [pcatrain{kk} datasample(tdd_feature{kk},sample_num,2)]; 
            %             size(pcatrain)
                        gmmtrain{kk} = [gmmtrain{kk} datasample(tdd_feature{kk},sample_num,2)];  
                    end
                catch
                    disp(filename);
                end
            end
        end
        save(['temp/pcatrain','_p_',num2str(pca_sample),'_d_',num2str(dim),'.mat'],'pcatrain');
        save(['temp/gmmtrain','_p_',num2str(pca_sample),'_d_',num2str(dim),'.mat'],'gmmtrain');
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

if ~exist(pca_gmm)
    [fU,fmu,fmeans, fcovariances, fpriors] = extract_pca(path_tdd,dim,num,pca_sample);
    save(pca_gmm,'fU','fmu','fmeans', 'fcovariances', 'fpriors');
end
temp = load(pca_gmm);
[U,mu,means, covariances, priors] = deal(temp.fU,temp.fmu,temp.fmeans,temp.fcovariances,temp.fpriors);

tic;

folderlist = getFolderList(path_tdd);
display('fv started ...');

for cfoldername=folderlist
    foldername=cfoldername{:};
    display(['proceeding ', foldername,'...']);
    if ~exist(fullfile(fv_dir,foldername),'dir')
        mkdir(fullfile(fv_dir,foldername));
    end
    
    filelist = getFolderList(fullfile(path_tdd,foldername,'*.mat'));
    for ctddfile = filelist
        tddfile = ctddfile{:};
%         if strcmp(tddfile,'v_shooting_10_03.mat')
%             display('stop')
%         end
        
        fv_file = fullfile(fv_dir,foldername,[tddfile(1:end-4),'.mat']);
        if exist(fv_file)
            log_exist = [log_exist,fv_file,char(13,10)'];
            continue;
        end
        
        try    
            tddfeature = load(fullfile(path_tdd,foldername,tddfile));
            tddfeature = tddfeature.idt_cnn_feature;
            for i=linspace(1,4,4)    
                tddfeature_pca = pcaApply(tddfeature{i},U{i},mu{i},dim); 
                fvfeat{i} = vl_fisher(single(tddfeature_pca),single(means{i}),single(covariances{i}),single(priors{i}));
                % pwer-L2 norm
                % L2 norm
                fvfeat{i} = sign(fvfeat{i}).*sqrt(abs(fvfeat{i}));
                fvfeat{i} = bsxfun(@rdivide,fvfeat{i},eps+sqrt(sum(fvfeat{i}.^2)));
    % 			gmmtrain = [gmmtrain datasample(tdd_feature_spatial_conv4_norm_1,6,2) datasample(tdd_feature_spatial_conv4_norm_2,6,2) datasample(tdd_feature_spatial_conv5_norm_1,6,2) datasample(tdd_feature_spatial_conv5_norm_2,6,2) ]; 
            end
            save(fv_file,'fvfeat');
        catch
            log_error = [log_error,tddfile,char(13,10)'];
        end
    end     
end
toc;
fprintf(fid,'%s\n%s',log_exist,log_error);
fclose(fid);
end


