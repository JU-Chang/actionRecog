
function [feature] = TDD(inf,tra,cnn_feature,scale_x,scale_y,num_cell)
    % TDD: perform trajectory pooling over convolutional feature maps.
    % Input:
    %       inf: information of trajectories from iDTs (10*N)
    %       traj: extracted trajectories (2L*N)
    %       cnn_feature: cnn feature maps (convlutional layers: W*H*C*L)
    %       scale_x: width ratio
    %       scale_y: height ratio
    %       num_cell: the number of cell in temporal dimension；a input
    %       parameter when extract iDT.
    % Output:
    %       feature: trajectory pooled descriptors ((C*NUM_CELL) *N)

    % choose only one scale of iDT.
    if ~isempty(inf)
        ind = inf(7,:)==1;
        inf = inf(:,ind);
        tra = tra(:,ind);
    end


    if ~isempty(inf)
        NUM_DIM = size(cnn_feature,3);
        NUM_DES = size(inf,2);
        TRA_LEN = size(tra,1)/2;

        num_fea = TRA_LEN / num_cell;

        % why -1?
        pos = reshape(tra,2,[])-1;
        % scale pos by scale_x,scale_y
        pos = round(bsxfun(@rdivide,pos,[scale_x;scale_y]) + 1);
        % get rid of data<1
        pos = bsxfun(@max,pos,[1;1]);
        % get rid of data out of the field of feature map.
        pos = bsxfun(@min,pos,[size(cnn_feature,2);size(cnn_feature,1)]);
        pos = reshape(pos,TRA_LEN*2,[]);

        cnn_feature = permute(cnn_feature,[1,2,4,3]);
        offset = [TRA_LEN-1:-1:0];
        size_mat = [size(cnn_feature,1),size(cnn_feature,2),size(cnn_feature,3)];
        cnn_feature = reshape(cnn_feature,[],NUM_DIM);

        cur_x = pos(1:2:end,:);           
        cur_y = pos(2:2:end,:);
        cur_t = bsxfun(@minus,inf(1,:),offset'); % fram_num of tra; inf(1,:) is the end frame of tra; 

        % cur_x:[15,:]
        % cur_y:[15,:]
        % cur_t:[15,:]  -->[frame_num-14,frame_num-13,...,frame_num-1; ...]

        % sub2ind:get the bilinear index
        % e.g.for a matrix A of size (2,3),element A(1,1)=A(1),A(2,1)=A(2)
        % so you get sub2ind([2,3],[1,2],[1,1]) : [1，2]
        tmp = cnn_feature(sub2ind(size_mat,cur_y,cur_x,cur_t),:)';
        tmp = reshape(tmp,NUM_DIM,num_fea,[]);
        % sum up every 15 frames.
        feature = reshape(sum(tmp,2),[],NUM_DES);
    else
        feature = [];
    end


end