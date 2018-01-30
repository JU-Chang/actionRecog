function split_hmdb51_script()
    % ############################################### %
    % configure
    split_dir = '/home/civic.org.cn/zyz/md128/HMDB51/testTrainMulti_7030_splits';
    split_type = 1;
    % ############################################### %

    % para.
    train_split = ['/home/civic.org.cn/zyz/md128/HMDB51/hmdb51_train_split',num2str(split_type),'.txt'];
    test_split = ['/home/civic.org.cn/zyz/md128/HMDB51/hmdb51_test_split',num2str(split_type),'.txt'];

    if ~exist(split_dir,'dir')          % check dir validation
        error(['split dir:',split_dir,' not exist!']); 
    end
    
    data = deal({{},{}});
%     data = deal({[],[]});
% 
%     % log_content = [log_content,'classname coding:',char(13,10)'];
    folderlist = dir(fullfile(split_dir,['*test_split',num2str(split_type),'.txt']));
    folderlist = {folderlist(:).name};
    for i=1:length(folderlist)
        split_file=folderlist{i};
    %     log_content = [log_content,foldername,' ',num2str(i), char(13,10)'];
        classname = split_file(1:end-16);
        fsplit = fopen(fullfile(split_dir,split_file));
        splitline = fgetl(fsplit);
        while ischar(splitline)
            splitline_ = strsplit(splitline);
            videoname = splitline_{1};
            istrain = str2num(splitline_{2});
            if (istrain)
                data{istrain}{end+1} = [classname,'/',videoname(1:end-4),char(13,10)'];
%                 data{istrain} = [data{istrain},classname,'/',videoname(1:end-4),char(13,10)'];

            end
            splitline = fgetl(fsplit);
        end 
        fclose(fsplit);
    end

    rdata1 = data{1}(randperm(length(data{1})));
%     disp(size(rdata1))
%     disp(class(rdata1))
    rdata1 = cell2mat(rdata1);
    rdata2 = data{2}(randperm(length(data{2})));
%     disp(size(rdata2))
%     disp(class(rdata2))

    rdata2 = cell2mat(rdata2);
    fts = fopen(train_split,'w');
    fprintf(fts,rdata1);
%     fprintf(fts,data{1});
    fclose(fts);
    fts2 = fopen(test_split,'w');
    fprintf(fts2,rdata2);
%     fprintf(fts2,data{2});
    fclose(fts2);
    disp('done');
end

