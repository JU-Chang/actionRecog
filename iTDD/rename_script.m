% rename some video file of hmdb51 containing #
logfile = '/home/civic.org.cn/zyz/md128/HMDB51/hmdb51_old_name.log';
fid = fopen(logfile,'r');
file_dir = fgetl(fid);
file_dir = strrep(file_dir,'/home/civic.org.cn/zyz/md128/HMDB51/video','/home/civic.org.cn/zyz/md128/HMDB51/tra_dir');
file_dir = strrep(file_dir,'.avi','.bin');
while ischar(file_dir)
    system(['rename \# s ',file_dir]);
    file_dir = fgetl(fid);
    file_dir = strrep(file_dir,'/home/civic.org.cn/zyz/md128/HMDB51/video','/home/civic.org.cn/zyz/md128/HMDB51/tra_dir');
    file_dir = strrep(file_dir,'.avi','.bin');
end
