% ExtractSpatialCNNFeature_hmdb51('spatial',3,0)

function ExtractSpatialCNNFeature_hmdb51(tag,scale,gpu_id)
    
    % LD_LIBRARY_PATH = ['/usr/local/MATLAB/R2016a/sys/opengl/lib/glnxa64:'...
    % '/usr/local/MATLAB/R2016a/sys/os/glnxa64:/usr/local/MATLAB/R2016a/bin/'...
    % 'glnxa64:/usr/local/MATLAB/R2016a/extern/lib/glnxa64:/usr/local/MATLAB/'...
    % 'R2016a/runtime/glnxa64:/usr/local/MATLAB/R2016a/sys/java/jre/glnxa64/'...
    % 'jre/lib/amd64/native_threads:/usr/local/MATLAB/R2016a/sys/java/jre/'...
    % 'glnxa64/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64:/usr/lib/:'];
    % setenv('LD_LIBRARY_PATH',LD_LIBRARY_PATH);  
%     addpath /home/chang/caffe-action_recog/matlab/
    % input:
    %       tag: 'spatial' or 'temporal'
    %       scale: a number from 1 to 

  % ################################################### %
    feat_path = ['/home/civic.org.cn/zyz/md128/HMDB51/',tag,'CnnFeature'];
    video_path = '/home/civic.org.cn/zyz/md128/HMDB51/video';
    log_file = ['/home/civic.org.cn/zyz/md128/HMDB51/',tag,'CnnFeature.log'];
  % ################################################### %

    
    model_def_file = [ 'model_proto/',tag,'_net_scale_',num2str(scale),'.prototxt'];
    model_file = [tag,'_v2.caffemodel'];
    sizes =[480,640; 340,454; 240,320; 170,227; 120,160];

    if ~exist(video_path,'dir')              % check dir validation
        error(['Video dir:"',video_path,'" not exist!']); 
    end
    
    fid = fopen(log_file,'w');
    fprintf(fid,'%s\n',datestr(now,0));
    log_exist = ['exist file:',char(13,10)'];
    log_error = ['error file:',char(13,10)'];
    
    if ~exist(feat_path,'dir')
        mkdir(feat_path);
%     elseif length(dir(feat_path)) > 2                 % check dir validation
%         error(['Feature file:"',feat_path,'" already exist!']); 
    end
    
    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    net = caffe.Net(model_def_file, model_file, 'test');
%     % spatial or temporal
%     if strcmp(tag,'spatial') 
%         layer1 = '4';
%         layer2 = '5';
%     %    data_list = video_list;
%     else
%         layer1 = '3';   
%         layer2 = '4';
%     %    data_list = flow_list;
%     end
% 
%     path4 = fullfile(feat_path,[tag,'_scale_',num2str(scale),'_conv',layer1]);
%     path5 = fullfile(feat_path,[tag,'_scale_',num2str(scale),'_conv',layer2]);

    function folderlist = getFolderList(ddir)
        folderlist = dir(ddir);
        folderlist = {folderlist(:).name};
        folderlist = setdiff(folderlist,{'.','..'});
    end

    tic;
    classlist = getFolderList(video_path);
    for i = 1:length(classlist)
        video_dir = fullfile(video_path,classlist{i});
        feat_dir = fullfile(feat_path,classlist{i});

        display(['processing ',classlist{i},'...']);
        % build dir of one class
        if ~exist(feat_dir,'dir')
            mkdir(feat_dir);     
        end

        % get filename of video
        videolist = getFolderList(fullfile(video_dir,'*.avi'));
        if isempty(videolist)
            continue
        end
        for k = 1:length(videolist)
            videofile = fullfile(video_dir,videolist{k});
            feat_file = fullfile(feat_dir,videolist{k});
            feat_file = [feat_file(1:end-4),'.mat'];
            
            if exist(feat_file)
                log_exist = [log_exist,feat_file,char(13,10)'];
                continue;
            end
            % do..
            try
                [feature_c5, feature_c4] = SpatialCNNFeature(videofile, net, sizes(scale,1), sizes(scale,2));
                cnnfeature{1} = feature_c4;
                cnnfeature{2} = feature_c5;
                save(feat_file,'cnnfeature');
            catch
                log_error = [log_error,videofile,char(13,10)'];
            end

        end
    end
    toc;
    fprintf(fid,'%s\n%s',log_exist,log_error);
    fclose(fid);
    
end