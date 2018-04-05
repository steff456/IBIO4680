%train
load('./eval_tools_Lab9/ground_truth/wider_face_train.mat')

%list of training image names
trainImages = {};
%Training boxes
trainBoxes = zeros(4,1);
%Images of the box
trainBoxImages = {};
%train box labels of targetclass
trainBoxLabels = [];
%train box patches
trainBoxPatches = {};
for i = 1: size(face_bbx_list,1)
    for j = 1: size(face_bbx_list{i}, 1)
        trainImages{end+1,1} = horzcat(file_list{i}{j}, '.jpg');
        for k = 1: size(face_bbx_list{i}{j},1)
            trainBoxes(:, end+1) = face_bbx_list{i}{j}(k,:)';
            trainBoxImages{end+1,1} = horzcat(file_list{i}{j}, '.jpg');
            splitt = strsplit(trainBoxImages{end}, '_');
            route_name = fullfile('TrainCrops',horzcat(splitt{1},'--',splitt{2}));
            dir_routes = dir(route_name);
            for l = 1:size(dir_routes,1)
                file_act = dir_routes(l);
                %disp(file_act.name)
                %disp(splitt{end})
                %disp('----------------')
                temp = strsplit(splitt{end},'.');
                filter_im = regexp(file_act.name, horzcat('im',temp{1},'c[0-9]+'));
                if filter_im == 1
                    disp('find one!')
                    im_act = imread(fullfile(route_name,  file_act.name));
                    trainBoxPatches{end+1,1} = im_act;
                end
            end
            trainBoxLabels(end+1,1) = str2double(splitt{1});
        end
        fprintf('i: %d - j: %d',i,j)
    end
end
trainBoxes = trainBoxes(:,2:end);
