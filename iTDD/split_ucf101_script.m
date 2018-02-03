function split_ucf101_script(split_type)
    % ############################################### %
    % configure
    split_dir = '/data/UCF101/ucfTrainTestlist';
    % ############################################### %

    % para.
    train_split_org = ['/data/UCF101/ucfTrainTestlist/trainlist0',num2str(split_type),'.txt'];
    train_split = ['/data/UCF101/ucfTrainTestlist/trainlist',num2str(split_type),'.txt'];

    if ~exist(split_dir,'dir')          % check dir validation
        error(['split dir:',split_dir,' not exist!']); 
    end
    
    data = {};
    fts = fopen(train_split_org);
    splitline = fgetl(fts);
    while ischar(splitline)
        splitline_ = strsplit(splitline);
        data{end+1} = [splitline_{1},char(13,10)'];
        splitline = fgetl(fts); 
    end
    fclose(fts);
    
    rdata1 = data(randperm(length(data)));
    rdata1 = cell2mat(rdata1);
    
    fts2 = fopen(train_split,'w');
    fprintf(fts2,rdata1);
    fclose(fts2);
    disp('done');
end