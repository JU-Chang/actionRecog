% extract_tdd_ucf101(3, 'spatial')
function extract_tdd_ucf101(scale,tag)
    
    % ############################################### %
    % configure
    data_dir = '/data/UCF101';
    log_file = ['/data/UCF101_',tag,'_tdd.log'];
    % ############################################### %
    
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
        error(['tra dir:',path_tra,' or feature dir:',path_feat,' not exist!']); 
    end
    
    if ~exist(save_path,'dir')
        mkdir(save_path);
    end
    
    fid = fopen(log_file,'w');
    fprintf(fid,'%s\n',datestr(now,0));
    log_exist = ['exist file:',char(13,10)'];
    log_error = ['error file:',char(13,10)'];
    
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
            
            tdd_file = fullfile(save_path,foldername{i},[filelist(j).name(1:end-4),'.mat']);
            if exist(tdd_file)
                log_exist = [log_exist,tdd_file,char(13,10)'];
                continue;
            end
            
            try
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
                        [norm_feature1, norm_feature2] = FeatureMapNormalization(cnnfeature{k});
                    
                        idt_cnn_feature{2*k-1} = TDD(info, tra, norm_feature1, sizes(scale,1), sizes(scale,2), 1);
                        idt_cnn_feature{2*k} = TDD(info, tra, norm_feature2, sizes(scale,1), sizes(scale,2), 1);
        %                 toc;
                    end
                    save(fullfile(save_path,foldername{i},[filelist(j).name(1:end-4),'.mat']),'idt_cnn_feature');
                end
            catch
                log_error = [log_error,tdd_file,char(13,10)'];
            end
        end
    end
    toc;
    fprintf(fid,'%s\n%s',log_exist,log_error);
    fclose(fid);
end

