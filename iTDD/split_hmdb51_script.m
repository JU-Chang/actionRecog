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

    data = deal({[],[]});

    tic;

    % log_content = [log_content,'classname coding:',char(13,10)'];
    folderlist = dir(fullfile(split_dir,['*test_split',num2str(split_type),'.txt']));
    folderlist = {folderlist(:).name;
    length(folderlist)
%     for i=1:length(folderlist)
%         split_file=folderlist{i};
%     %     log_content = [log_content,foldername,' ',num2str(i), char(13,10)'];
%         classname = split_file(1:end-16);
%         fsplit = fopen(split_file);
%         splitline = fgetl(fsplit);
%         while ischar(splitline)
%             splitline_ = strsplit(splitline);
%             videoname = splitline_{1};
%             istrain = str2num(splitline_{2});
%             if (istrain)
%                 data{istrain} = [data{istrain},classname,'/',videoname];
%             end
%             splitline = fgetl(fsplit);
%         end 
%         fclose(fsplit);
%     end

    fid = fopen(train_split,'w');
    fprintf(fid,data{1});
    fclose(fid);
    fid = fopen(test_split,'w');
    fprintf(fid,data{2});
    fclose(fid);
end

