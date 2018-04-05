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
            trainBoxLabels(end+1,1) = str2double(splitt{1});
            route =  
            trainBoxPatches{end+1} = im_act;
        end
        j
    end
end
trainBoxes = trainBoxes(:,2:end);