clc;clearvars;

data=load('C:\Users\Srikanth\Documents\MATLAB\MLBP\eX 6\auto_mpg_dataset\auto-mpg.csv');

labels = importdata('C:\Users\Srikanth\Documents\MATLAB\MLBP\eX 6\auto_mpg_dataset\auto-mpg-names.txt');

% Covariance of the data

meanData = sum(data); % sum the data feature wise 1x8 matrix
meanData= meanData/size(data,1);
repmat_sampleMean = repmat(meanData,398,1);
first  = (data - repmat_sampleMean);
dataCovariance = (first' * first ) / (size(data,1));

%dataCovariance = cov(data);


% Eigen Value decomposition of Covariance Matrix

[V,D,W] = eig(dataCovariance); % eig gives the eigen values for cov matrix.

%% Output 

output = (V(:,7:8)' * first')';

figure; scatter(output(:,1),output(:,2))

figure; scatter(output(:,1),output(:,2));text(output(:,1),output(:,2),labels)

output1 = (V' * first')';

amountVariance = sum(var(output))/sum(var(output1));

disp(['The amount of variance explanined by first two components is ', num2str(amountVariance*100), ' %'])












