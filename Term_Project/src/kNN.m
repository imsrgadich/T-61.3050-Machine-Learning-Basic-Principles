clc,clearvars;

%% Red - 1 and White - 01
data = load('training_classification_regression_2015.csv');
labels = data(:,13);
%regression_output = data(:,12);
data = data(:,1:11);

[m, n] = size(data);

%% Define a k here
k =2;

% Mean Normalize data - subtract with mean and divide with max in each col

meanData = sum(data); % sum the data feature wise 1x8 matrix
meanData= meanData/size(data,1);
repmat_sampleMean = repmat(meanData,m,1);
data = data - repmat_sampleMean;
data  = data./repmat(std(data),m,1);

%% On disrect data - 80% train - 20% validation
data_train = data(1:4000,:);
[m_train, n_train] = size(data_train);
labels_train = labels(1:4000);
data_val = data(4001:5000,:);
labels_val = labels(4001:5000);
[m_val, n_val] = size(data_val);

labels_knn = zeros(m_val,1);
%% Algo starts
for i = 1:m_val
    
    distance = sqrt((sum(((data_train - repmat(data_val(i,:),m_train,1)).^2),2)));
    [dist_sort,dist_index] = sort(distance);
    dist_index = dist_index(1:k);
    labels_knn(i,1) = mode(labels_train(dist_index));
        
end

error = 100 - 100 * (1- sum(labels_val ~= labels_knn)/m_val);
accuracy = 100 - error;
