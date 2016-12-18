clc,clearvars;

%% Red - 1 and White - 01
data = load('training_classification_regression_2015.csv');
labels = data(:,13);

%% Leaving out 7th feature
data = [data(:,1:6) data(:,7:11)];
dataKaggle = load('challenge_public_test_classification_regression_2015.csv');
dataKaggle = dataKaggle(:,2:12);
dataKaggle = [dataKaggle(:,1:6) dataKaggle(:,7:11)];

[m, n] = size(data);
[mK, nK] = size(dataKaggle);

%% Define a k here
k =13;

% Standarize data - subtract with mean and divide with max in each col

meanData = mean(data); % sum the data feature wise 1x8 matrix
repmat_sampleMean = repmat(meanData,m,1);
data = data - repmat_sampleMean;
stdData = std(data);
data  = data./repmat(stdData,m,1);

%% Standarize the kaggle data with the same statistics of training data

repmat_sampleMeanK = repmat(meanData,mK,1);
dataKaggle = dataKaggle - repmat_sampleMeanK;
dataKaggle  = dataKaggle./repmat(stdData,mK,1);


%% On disrect data - 80% train - 20% validation
data_train = data;
[m_train, n_train] = size(data_train);
labels_train = labels;
data_val = dataKaggle;
%labels_val = labels(4001:5000);
[m_val, n_val] = size(data_val);

labels_knn = zeros(m_val,1);
%% Algo starts
for i = 1:m_val
    
    distance = sqrt((sum(((data_train - repmat(data_val(i,:),m_train,1)).^2),2)));
    [dist_sort,dist_index] = sort(distance);
    dist_index = dist_index(1:k);
    labels_knn(i,1) = mode(labels_train(dist_index));
        
end

%error = 100 - 100 * (1- sum(labels_val ~= labels_knn)/m_val);
%accuracy = 100 - error;
 a=load('99%.csv');
sum(a ~= labels_knn)