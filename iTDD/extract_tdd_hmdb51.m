% extract_tdd_hmdb51(3, 'spatial')
function extract_tdd_hmdb51(scale,tag)
    
    % configure
    data_dir = '/data/HMDB51/';
    
%     if strcmp(tag,'spatial')
%         layer1 = 'conv4';
%         layer2 = 'conv5';
%     elseif scmp(tag,'temporal')
%         layer1 = 'conv3';
%         layer2 = 'conv4';
%     else 
%         error('wrong input!')
%     end
    
    path_tra = fullfile(data_dir,'tra_dir');
    path_feat = fullfile(data_dir,[tag,'CnnFeature']);
    save_path = fullfile(data_dir,['tdd_',tag,'_scale_',num2str(scale)]);
    sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
    
    if ~exist(path_tra,'dir') || ~exist(path_feat,'dir')              % check dir validation
        error(['tra dir:',video_path,' or feature dir:',path_feat,' not exist!']); 
    end
    
    if ~exist(save_path,'dir')
        mkdir(save_path);
    elseif length(dir(save_path)) > 2                 % check dir validation
        error(['Feature file:"',save_path,'" already exist!']); 
    end
    
    folderlist = dir(path_tra);
    foldername = {folderlist(:).name};
    foldername = setdiff(foldername,{'.','..'});

    tic;
    for i = 1:length(foldername)
        if ~exist(fullfile(save_path,foldername{i}),'dir')
            mkdir(fullfile(save_path,foldername{i}));
        end
        display(['processing ',foldername{i},'...']);

        filelist = dir(fullfile(path_tra,[foldername{i},'/*.bin']));
        for j = 1:length(filelist)
            
            tra_file = fullfile(path_tra,foldername{i},[filelist(j).name(1:end-4),'.bin']);
            
            data = import_idt(tra_file);
            info = data.info;
            tra = data.tra;
            if ~isempty(info)
%                 tic;
                feat_file = fullfile(path_feat,foldername{i},[filelist(j).name(1:end-4),'.mat']);
                if ~exist(feat_file,'file')
                    error('feature files not exist.')
                end
                f = load(feat_file);
                cnnfeature = f.cnnfeature;
                for k =[1,2]

                    if max(info(1,:)) > size(cnnfeature{k},4)
                        ind =  info(1,:) <= size(cnnfeature{k},4);
                        info = info(:,ind);
                        tra = tra(:,ind);
                    end
                    [norm_feature1, norm_feature2] = FeatureMapNormalization(feature);
                    idt_cnn_feature{2*k-1} = TDD(info, tra, norm_feature1, sizes(scale,1), sizes(scale,2), 1);
                    idt_cnn_feature{2*k} = TDD(info, tra, norm_feature2, sizes(scale,1), sizes(scale,2), 1)
    %                 toc;
                end
                save(fullfile(save_path,foldername{i},[filelist(j).name(1:end-4),'.mat']),'idt_cnn_feature');
            end
        end
    end
    toc;
end

