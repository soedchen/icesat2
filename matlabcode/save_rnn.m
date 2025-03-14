clc;clearvars;
diff_location = [{'huaguangjiao'}];

for i = 1 :1%length(diff_location)
    allpath = ['H:\codes\image\full\'];

    train_path =['G:\thesis5\st_cp_rrs_train2\'];
    mkdir(train_path);
    %训练网络
    savename = [train_path,'all_train_depth_3.mat'];
    load(savename,'depth_all')
    savename = [train_path,'all_train_rrs_3.mat'];
    load(savename,'rrs_all')

    train_rrs = rrs_all';
    train_depth = depth_all';

    net = rnntrain(train_rrs,train_depth);
    net_name = [net_path,'net_virgin.mat'];
    save(net_name,'net');
    
     for j = 1:length(subsetFiles)
        date1 = subsetFiles{j};
        load([partpath,date1]);
        train_data2= A;
        [~,~,lege] = size(train_data2);
        train_data = reshape(train_data2,[size(train_data2,1)*size(train_data2,2),lege]);
        disp('training data collected')
        load([fullpath,date1]);
        test_data2= A;
        test_data = reshape(test_data2,[size(test_data2,1)*size(test_data2,2),lege]);
        disp('testing data collected')

        %%定义网络
        Predict_label1 = nnfit(test_data',train_data',train_depth');
        depth_nn_50=reshape(Predict_label1',size(LON));%

        %存储结果
        [~,R] = readgeoraster([fullpath,date1(1:end-4),'.tif']);
        OutfileName = [results_path,date1(1:end-4),'.tif'];
        geotiffwrite(OutfileName,depth_nn_50, R);

    end

end