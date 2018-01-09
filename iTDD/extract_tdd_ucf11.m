% extract_tdd_ucf11(3, 'spatial')
function extract_tdd_ucf11(scale,tag)
    
    % configure
    data_dir = '/data/UCF11/';
    
    if strcmp(tag,'spatial')
        layer1 = 'conv4';
        layer2 = 'conv5';
    elseif scmp(tag,'temporal')
        layer1 = 'conv3';
        layer2 = 'conv4';
    else 
        error('wrong input!')
    end
    
    path_tra = fullfile(data_dir,'tra_dir');
    path_feat = fullfile(data_dir,[tag,'CnnFeature']);
    save_path{1} = fullfile(data_dir,['tdd_',tag,'_scale_',num2str(scale),'_',layer1]);
    save_path{2}= fullfile(data_dir,['tdd_',tag,'_scale_',num2str(scale),'_',layer2]);
    
    sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];

    folderlist = dir(path_tra);
    foldername = {folderlist(:).name};
    foldername = setdiff(foldername,{'.','..'});

    tic;
    for i = 1:length(foldername)
        if ~exist(fullfile(save_path{1},foldername{i}),'dir')
            mkdir(fullfile(save_path{1},foldername{i}));
        else
            error('target file exist!')
        end
        if ~exist(fullfile(save_path{2},foldername{i}),'dir')
            mkdir(fullfile(save_path{2},foldername{i}));
        else
            error('target file exist!')
        end
        display(['processing ',foldername{i},'...']);

        filelist = dir(fullfile(path_tra,[foldername{i},'/*.bin']));
        for j = 1:length(filelist)
            
            tra_file = fullfile(path_tra,foldername{i},[filelist(j).name(1:end-4),'.bin']);
            if ~exist(tra_file,'file')
                error('trajectory files not exist.')
            end
            
            data = import_idt(tra_file);
            info = data.info;
            tra = data.tra;
            if ~isempty(info)
%                 tic;
                feat_folder{1} = [tag,'_scale_',num2str(scale),'_',layer1];
                feat_folder{2} = [tag,'_scale_',num2str(scale),'_',layer2];
                
                for k =[1,2]
                    feat_file = fullfile(path_feat,feat_folder{k},foldername{i},[filelist(j).name(1:end-4),'.mat']);
                    if ~exist(feat_file,'file')
                        error('trajectory files not exist.')
                    end
                    f = load(feat_file);
                    feature = f.feature;
                    if max(info(1,:)) > size(feature,4)
                        
                        ind =  info(1,:) <= size(feature,4);
                        info = info(:,ind);
                        tra = tra(:,ind);
                    end
                    [cnn_feature1, ~] = FeatureMapNormalization(feature);
                    idt_cnn_feature = TDD(info, tra, cnn_feature1, sizes(scale,1), sizes(scale,2), 1);
                    save(fullfile(save_path{k},foldername{i},[filelist(j).name(1:end-4),'.mat']),'idt_cnn_feature');
    %                 toc;
                end
            end
        end
    end
    toc;
end

