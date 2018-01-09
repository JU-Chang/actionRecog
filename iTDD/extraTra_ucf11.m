% iDT extraction of ucf11.
% @author=zz

function extraTra_ucf11()


% path1 = getenv('LD_LIBRARY_PATH');             %获得系统路径的字符串
% path1 = ['/usr/lib:' path1];   %字符串中加入自己要的路径
% % path1 = [path1 '::/usr/local/ffmpeg/bin  /home/chang/anaconda2/bin/']   %字符串中加入自己要的路径
% setenv('LD_LIBRARY_PATH', path1);                %设置系统路径
% %输出以查看之
% !echo $PATH                              

data_dir = '/data/UCF11/action_youtube_naudio';
tra_dir = '/data/UCF11/tra_dir';

if ~exist(tra_dir,'dir')
    mkdir(tra_dir);
elseif length(dir(tra_dir)) > 2                 % check dir validation
    error(['trajecotry:"',tra_dir,'" already exist!']); 
end

if ~exist(data_dir,'dir')              % check dir validation
    error(['data dir:"',data_dir,'" not exist!']); 
end

function folderlist = getFolderList(ddir)
    folderlist = dir(ddir);
    folderlist = {folderlist(:).name};
    folderlist = setdiff(folderlist,{'.','..'});
end


classlist = getFolderList(data_dir);
for i = 1:length(classlist)
    video_dir = fullfile(data_dir,classlist{i});
    vtra_dir = fullfile(tra_dir,classlist{i});
    if ~exist(vtra_dir,'dir')
        mkdir(vtra_dir);
        display(['processing',vtra_dir]);
    else
        error('target file exist!')
    end
    videofolderlist = getFolderList(video_dir);
    for j = 1:length(videofolderlist)
        video_dir2 = fullfile(video_dir,videofolderlist{j});
        videolist = getFolderList(fullfile(video_dir2,'*.avi'));
        if isempty(videolist)
            continue
        else
            for k = 1:length(videolist)
                videofile = fullfile(video_dir2,videolist{k});
                vtra_file = fullfile(vtra_dir,videolist{k});
                vtra_file = [vtra_file(1:end-4),'.bin'];
                %display('Extract improved trajectories...');
                system(['LD_LIBRARY_PATH=/usr/lib /home/chang/code/actionRecog/iTDD/DenseTrackStab -f ',videofile,' -o ',vtra_file]);
            end
        end
    end
end

display('done');




% iDT extraction
% display('Extract improved trajectories...');
% function  extract_fv(index, power, layer, tag ,dim1, norm)
% system(['./DenseTrackStab -f ',vid_name,' -o ',vid_name(1:end-4),'.bin']);


end
