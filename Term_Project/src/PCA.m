clc,clearvars;

%% Red - 1 and White - 01
data = load('training_classification_regression_2015.csv');
labels = data(:,13);
regression_output = data(:,12);
data = data(:,1:11);

[m, n] = size(data);

% Covariance of the data

meanData = sum(data); % sum the data feature wise 1x8 matrix
meanData= meanData/size(data,1);
repmat_sampleMean = repmat(meanData,m,1);
first  = (data - repmat_sampleMean);
dataCovariance = (first' * first ) / (size(data,1));

%dataCovariance = cov(data);


% Eigen Value decomposition of Covariance Matrix

[V,D,W] = eig(dataCovariance); % eig gives the eigen values for cov matrix.

%% Output 

output = (V(:,7:11)' * first')';

figure;
scatter(output(labels==1,1),output(labels==1,2),'o');hold on
scatter(output(labels==0,1),output(labels==0,2),'o');hold off
figure;
scatter(output(labels==1,1),output(labels==1,3),'o');hold on
scatter(output(labels==0,1),output(labels==0,3),'o');hold off
figure;
scatter(output(labels==1,1),output(labels==1,4),'o');hold on
scatter(output(labels==0,1),output(labels==0,4),'o');hold off
figure;
scatter(output(labels==1,1),output(labels==1,4),'o');hold on
scatter(output(labels==0,1),output(labels==0,4),'o');hold off
figure;
scatter(output(labels==1,1),output(labels==1,5),'o');hold on
scatter(output(labels==0,1),output(labels==0,5),'o');hold off
figure;
scatter(output(labels==1,2),output(labels==1,3),'o');hold on
scatter(output(labels==0,2),output(labels==0,3),'o');hold off
figure;
scatter(output(labels==1,2),output(labels==1,4),'o');hold on
scatter(output(labels==0,2),output(labels==0,4),'o');hold off
figure;
scatter(output(labels==1,2),output(labels==1,5),'o');hold on
scatter(output(labels==0,2),output(labels==0,5),'o');hold off
figure;
scatter(output(labels==1,3),output(labels==1,4),'o');hold on
scatter(output(labels==0,3),output(labels==0,4),'o');hold off
figure;
scatter(output(labels==1,3),output(labels==1,5),'o');hold on
scatter(output(labels==0,3),output(labels==0,5),'o');hold off
figure;
scatter(output(labels==1,4),output(labels==1,5),'o');hold on
scatter(output(labels==0,4),output(labels==0,5),'o');hold off

% output1 = (V' * first')';
% 
% amountVariance = sum(var(output))/sum(var(output1));
% 
% disp(['The amount of variance explanined by first 5 components is ', num2str(amountVariance*100), ' %'])

%========================================================================================================================%

classes = 2;
features = 5;

%% Separating the training inputs and targets!
trainingInput = output;
trainingTarget = labels;

%% Calulating the probability for Class 1 i.e., rt = 1 
probabilityClass2 = sum(trainingTarget(:) == 1)/size(trainingTarget,1);  
probabilityClass1 = 1 - probabilityClass2; % p(rt = 0) = 1 - p(rt = 1)   

%% Finding Pij's for all the features!

P = zeros(classes,features);
    
for j = 1: features
    
    featureColumn = trainingInput(:,j);
    
    P(2,j) = (1+sum(featureColumn(trainingTarget == 1)))/(2+sum(trainingTarget));
    P(1,j) = (1+sum(featureColumn(trainingTarget == 0)))/(2+sum(trainingTarget));
    
end

%% Weights and weightNot


weights = log((P(1,:).*(1-P(2,:)))./(P(2,:).*(1-P(1,:))));

[sortedWeights,sortedIndex] = sort(weights,'descend');

weightNot = sum(log((1-P(1,:))./(1-P(2,:)))) + log(probabilityClass1/probabilityClass2); 

%% Accuracy on Training Set

classification = (1./(1+exp(weightNot + weights * trainingInput'))) > 0.5;

accuracy = (size(trainingTarget,1) - sum(abs(classification' - trainingTarget)))/size(trainingTarget,1);

%disp(['The accuracy obtained for training set is ',num2str(accuracy*100),'%'])

disp(['The clasification error obtained for training set is ',num2str(100- accuracy*100),'%'])

%% ******************************************************************************** %%
%% Accuracy on Test Set

testClassification = (1./(1+exp(weightNot + weights * testData(:,2:end)'))) > 0.5;

testAccuracy = (size(testData,1) - sum(abs(testClassification' - testData(:,1))))/size(testData,1);

%disp(['The accuracy obtained for test set is ',num2str(testAccuracy*100),'%'])

disp(['The clasification error obtained for test set is ',num2str(100 - testAccuracy*100),'%'])











