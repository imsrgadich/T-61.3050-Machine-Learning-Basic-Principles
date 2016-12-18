clc,clearvars;

%% Red - 1 and White - 0
data = load('training_classification_regression_2015.csv');
data = [data(:,1:11) data(:,12)];

[m, n] = size(data);
meanData = mean(data); % sum the data feature wise 1x8 matrix
stdData = std(data);
repmat_sampleMean = repmat(meanData,m,1);
repmat_sampleStd = repmat(stdData,m,1);
dataMod  = (data - repmat_sampleMean)./repmat_sampleStd;

regression_output = dataMod(:,12);
data = dataMod(:,1:11);



% Covariance of the data


dataCovariance = (dataMod' * dataMod ) / (size(data,1));

%% On disrect data - 80% train - 20% validation
data_train = data(1:4000,:);
data_val = data(4001:5000,:);
weights = pinv(data_train'*data_train)*data_train' * regression_output(1:4000);
output_norm = round((data_val * weights)*stdData(12) + meanData(12));
errorTest = mean((regression_output(4001:5000) - output_norm).^2);
text = ['Direct Data - The validation mean squared error on validation set is ',num2str(errorTest)];
disp(text)

%% On mean normalized data 
data_train = dataMod(1:4000,:);
data_val = dataMod(4001:5000,:);
weights = pinv(data_train'*data_train)*data_train' * regression_output(1:4000);
output_mean = round(data_val * weights);
errorTest = mean((regression_output(4001:5000) - output_mean).^2);
text = ['Standarized Data - The validation mean squared error on validation set is ',num2str(errorTest)];
disp(text)

%% PCA data

[V,D,W] = eig(dataCovariance); % eig gives the eigen values for cov matrix.

%% Output 

data_mod = (V(:,7:11)' * dataMod')';
data_train = dataMod(1:4000,:);
data_val = dataMod(4001:5000,:);
weights = pinv(data_train'*data_train)*data_train' * regression_output(1:4000);
output_mean_pca = round((data_val * weights)*stdData(12) + meanData(12));
errorTest = mean((regression_output(4001:5000) - output_mean_pca).^2);
text = ['PCA Data - The validation mean squared error on validation set is ',num2str(errorTest)];
disp(text)


