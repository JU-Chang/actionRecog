% train_svm_ucf101(1)
function train_svm_ucf101(splitType)
    
    %addpath(genpath('/home/civic.org.cn/zyz/'));
    %input:
    %   splitType:[1,2,3],choose a split txt file

    % ############################################### %
    % configure
    fv_dir = '/data/UCF101/pnormfv_spatial_psam_6_dim_64';
    split_dir = '/data/UCF101/ucfTrainTestlist';
    log_dir = '/data/UCF101/svmTrainLog';
    model_dir = '/data/UCF101/svmModel';
    % ############################################### %

    % para.
    % dim = 64;
    % num = 256;
    % pca_sample = 100;
    if ~exist(fv_dir,'dir')
        error(['tdd dir:',fv_dir,' not exist!']);
    end

    if ~exist(split_dir,'dir')          % check dir validation
        error(['split dir:',split_dir,' not exist!']); 
    end

    if ~exist(log_dir,'dir')
        mkdir(log_dir);
    end

    if ~exist(model_dir,'dir')
        mkdir(model_dir);
    end

    % data = {[],[]};
    % label = {{},{}};

    train_split_file = fullfile(split_dir,['trainlist',num2str(splitType),'.txt']);
    test_split_file = fullfile(split_dir,['testlist0',num2str(splitType),'.txt']);
    log_file = fullfile(log_dir,[datestr(now,0),'svm_fv.log']);
    model_file = fullfile(model_dir,['svm_s',num2str(splitType),datestr(now,0),'.mat']);

    log_content = ['svm config:',char(13,10)'];
    log_content = [log_content,'splitType:',num2str(splitType),char(13,10)'];
    log_content = [log_content,'modelFile:',model_file,char(13,10)'];
    log_content = [log_content,'fv:',fv_dir,char(13,10)'];
    log_error = ['error file:',char(13,10)'];

    % function folderlist = getFolderList(ddir)
    %     folderlist = dir(ddir);
    %     folderlist = {folderlist(:).name};
    %     folderlist = setdiff(folderlist,{'.','..'});
    %     folderlist = folderlist;
    % end
    % 
    % tic;
    % 
    % % log_content = [log_content,'classname coding:',char(13,10)'];
    % folderlist = getFolderList(fv_dir);
    

    % for i = 1:length(folderlist)
    %     foldername=folderlist{i};
    % %     log_content = [log_content,foldername,' ',num2str(i), char(13,10)'];
    %     fsplit = fopen(fullfile(split_dir,[foldername,'_test_split',num2str(splitType),'.txt']));
    %     splitline = fgetl(fsplit);
    %     while ischar(splitline)
    %         splitline_ = strsplit(splitline);
    %         videoname = splitline_{1};
    %         fvfile = fullfile(fv_dir,foldername,[videoname(1:end-4),'.mat']);
    %         try
    %             fvfeat_ = load(fvfile);
    %             fvfeat = fvfeat_.fvfeat;
    %         catch
    %             log_error = [log_error,fvfile,char(13,10)'];
    %             splitline = fgetl(fsplit);
    %             continue;
    %         end
    %         istrain = str2num(splitline_{2});
    %         if (istrain)
    %             data{istrain} = [data{istrain};fvfeat{1}' fvfeat{2}' fvfeat{4}' fvfeat{4}'];
    %             label{istrain}{length(label{istrain})+1} = foldername;
    %         end
    %         splitline = fgetl(fsplit);
    %     end 
    %     fclose(fsplit);
    % end
    function [ddata,llabel,llog_error]= get_data(split_file,llog_error)

    %     log_content = [log_content,foldername,' ',num2str(i), char(13,10)'];
        fspt = fopen(split_file);
        splitline = fgetl(fspt);
        ddata = [];
        llabel = {};
        while ischar(splitline)
            splitline_ = strsplit(splitline,'/');
            videoname = splitline_{2}(1:end-4);
            foldername = splitline_{1};
            fvfile = fullfile(fv_dir,foldername,[videoname,'.mat']);
            try
                fvfeat_ = load(fvfile);
                fvfeat = fvfeat_.fvfeat;
            catch
                llog_error = [llog_error,fvfile,char(13,10)'];
                splitline = fgetl(fspt);
                continue;
            end
            ddata = [ddata;fvfeat{1}' fvfeat{2}' fvfeat{4}' fvfeat{4}'];
            llabel{length(llabel)+1} = foldername;
            splitline = fgetl(fspt); 
        end
        
        fclose(fspt);
    end
    
    tic
    disp('get data ...');
%     if ~exist('data.mat')
    [data{1},label{1},log_error] = get_data(train_split_file,log_error);
    [data{2},label{2},log_error] = get_data(test_split_file,log_error);
%         save('data.mat','data','label')
%     end
    
    svmModel = fitcecoc(data{1},label{1}');
    predictLabels = predict(svmModel,data{2});
    acc = strcmp(predictLabels,label{2}');
    acc = sum(acc)/length(acc);
    save(model_file,'svmModel');
    log_content = [log_content,'accuracy:',num2str(acc),char(13,10)'];

    display(['acc:',num2str(acc)]);
    toc;
    fid = fopen(log_file,'w');
    fprintf(fid,'%s\n',datestr(now,0));
    fprintf(fid,'%s\n%s',log_content,log_error);
    fclose(fid);
end