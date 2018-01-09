% ExtractSpatialCNNFeature_ucf11('spatial',3,0)

function ExtractSpatialCNNFeature_ucf11(tag,scale,gpu_id)

    % LD_LIBRARY_PATH = ['/usr/local/MATLAB/R2016a/sys/opengl/lib/glnxa64:'...
    % '/usr/local/MATLAB/R2016a/sys/os/glnxa64:/usr/local/MATLAB/R2016a/bin/'...
    % 'glnxa64:/usr/local/MATLAB/R2016a/extern/lib/glnxa64:/usr/local/MATLAB/'...
    % 'R2016a/runtime/glnxa64:/usr/local/MATLAB/R2016a/sys/java/jre/glnxa64/'...
    % 'jre/lib/amd64/native_threads:/usr/local/MATLAB/R2016a/sys/java/jre/'...
    % 'glnxa64/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64:/usr/lib/:'];
    % setenv('LD_LIBRARY_PATH',LD_LIBRARY_PATH);
    addpath /home/chang/caffe-action_recog/matlab/
    % input:
    %       tag: 'spatial' or 'temporal'
    %       scale: a number from 1 to 


    sizes =[480,640; 340,454; 240,320; 170,227; 120,160];
    % video_list =  '/data/UCF11/ucf11_train_split_2.txt';
    % flow_list =  '';
    feat_path = ['/data/UCF11/',tag,'CnnFeature'];
    video_path = '/data/UCF11/action_youtube_naudio';

    if ~exist(feat_path,'dir')
        mkdir(feat_path);
    elseif length(dir(feat_path)) > 2                 % check dir validation
        error(['Feature file:"',feat_path,'" already exist!']); 
    end

    if ~exist(video_path,'dir')              % check dir validation
        error(['Video dir:"',video_path,'" not exist!']); 
    end

    % spatial or temporal
    if strcmp(tag,'spatial') 
        layer1 = '4';
        layer2 = '5';
    %    data_list = video_list;
    else
        layer1 = '3';   
        layer2 = '4';
    %    data_list = flow_list;
    end

    path4 = fullfile(feat_path,[tag,'_scale_',num2str(scale),'_conv',layer1]);
    path5 = fullfile(feat_path,[tag,'_scale_',num2str(scale),'_conv',layer2]);

    model_def_file = [ 'model_proto/',tag,'_net_scale_',num2str(scale),'.prototxt'];
    model_file = [tag,'_v2.caffemodel'];
    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    net = caffe.Net(model_def_file, model_file, 'test');

    function folderlist = getFolderList(ddir)
        folderlist = dir(ddir);
        folderlist = {folderlist(:).name};
        folderlist = setdiff(folderlist,{'.','..'});
    end

    tic;
    classlist = getFolderList(video_path);
    for i = 1:length(classlist)
        video_dir = fullfile(video_path,classlist{i});
        feat_dir1 = fullfile(path4,classlist{i});
        feat_dir2 = fullfile(path5,classlist{i});

        display(['processing ',classlist{i},'...']);
        % build dir of one class
        if ~exist(feat_dir1,'dir')
            mkdir(feat_dir1);     
        else
            error('target file exist!')
        end

        if ~exist(feat_dir2,'dir')
            mkdir(feat_dir2);     
        else
            error('target file exist!')
        end

        videofolderlist = getFolderList(video_dir);
        for j = 1:length(videofolderlist)
            video_dir2 = fullfile(video_dir,videofolderlist{j});

            % get filename of video
            videolist = getFolderList(fullfile(video_dir2,'*.avi'));
            if isempty(videolist)
                continue
            else
                for k = 1:length(videolist)
                    videofile = fullfile(video_dir2,videolist{k});

                    % do..
    %                 tic;
                    [feature_c5, feature_c4] = SpatialCNNFeature(videofile, net, sizes(scale,1), sizes(scale,2));
    %                 toc;
    %                 tic;
                    feature = feature_c5;
                    feat_file = fullfile(path5,classlist{i},videolist{k});
                    feat_file = [feat_file(1:end-4),'.mat'];
                    save(feat_file,'feature');
                    feature = feature_c4;
                    feat_file = fullfile(path4,classlist{i},videolist{k});
                    feat_file = [feat_file(1:end-4),'.mat'];
                    save(feat_file,'feature');
    %                 toc;

                end
            end
        end
    end
    toc;
    caffe.reset_all();
end