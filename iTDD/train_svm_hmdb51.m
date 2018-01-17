% train_svm_hmdb51(1)
function train_svm_hmdb51(splitType)
    
%addpath(genpath('/home/civic.org.cn/zyz/'));
%input:
%   splitType:[1,2,3],choose a split txt file

% ############################################### %
% configure
fv_dir = '/home/civic.org.cn/zyz/md128/HMDB51/fv_spatial';
split_dir = '/home/civic.org.cn/zyz/md128/HMDB51/testTrainMulti_7030_splits';
log_dir = '/home/civic.org.cn/zyz/md128/HMDB51/svmTrainLog';
model_dir = '/home/civic.org.cn/zyz/md128/HMDB51/svmModel';
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

data = {[],[]};
label = {{},{}};

log_file = fullfile(log_dir,[datestr(now,0),'svm_fv.log']);
model_file = fullfile(model_dir,['svm_s',num2str(splitType),datestr(now,0),'.mat']);
fid = fopen(log_file,'w');
fprintf(fid,'%s\n',datestr(now,0));
log_content = ['svm config:',char(13,10)'];
log_content = [log_content,'splitType:',num2str(splitType),char(13,10)'];
log_content = [log_content,'modelFile:',model_file,char(13,10)'];
log_error = ['error file:',char(13,10)'];

function folderlist = getFolderList(ddir)
    folderlist = dir(ddir);
    folderlist = {folderlist(:).name};
    folderlist = setdiff(folderlist,{'.','..'});
    folderlist = folderlist;
end

tic;

% log_content = [log_content,'classname coding:',char(13,10)'];
folderlist = getFolderList(fv_dir);
display('svm training started ...');

for i = 1:length(folderlist)
    foldername=folderlist{i};
%     log_content = [log_content,foldername,' ',num2str(i), char(13,10)'];
    fsplit = fopen(fullfile(split_dir,[foldername,'_test_split',num2str(splitType),'.txt']));
    splitline = fgetl(fsplit);
    while ischar(splitline)
        splitline_ = strsplit(splitline);
        videoname = splitline_{1};
        fvfile = fullfile(fv_dir,foldername,[videoname(1:end-4),'.mat']);
        try
            fvfeat_ = load(fvfile);
            fvfeat = fvfeat_.fvfeat;
        catch
            log_error = [log_error,fvfile,char(13,10)'];
            splitline = fgetl(fsplit);
            continue;
        end
        istrain = str2num(splitline_{2});
        if (istrain)
            data{istrain} = [data{istrain};fvfeat{1}' fvfeat{2}' fvfeat{4}' fvfeat{4}'];
            label{istrain}{length(label{istrain})+1} = foldername;
        end
        splitline = fgetl(fsplit);
    end 
    fclose(fsplit);
end

svmModel = fitcecoc(data{1},label{1}');
predictLabels = predict(svmModel,data{2});
acc = strcmp(predictLabels,label{2}');
acc = sum(acc)/length(acc);
save(model_file,'svmModel');

display(['acc:',acc]);
toc;
fprintf(fid,'%s\n%s',log_content,log_error);
fclose(fid);
end