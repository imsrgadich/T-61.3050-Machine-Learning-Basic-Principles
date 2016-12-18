clc;clearvars;

data = load('training_classification_regression_2015.csv');
[m, n] = size(data);
features = 11;

dataTrain = data(1:4000,1:features); 
dataTest= data(4001:end,1:features);
dataTargetTrain= data(1:4000,13);
dataTargetTest = data(4001:end,13);
%% Note here that test data is normalized with the training data statistics
dataTrainMod = (dataTrain - mean(dataTrain(:)))./ mean(dataTrain(:));
dataTest = (dataTest - mean(dataTrain(:)))./ mean(dataTrain(:));
dataTrainMod = [ones(size(dataTrain,1),1) dataTrainMod];
dataTestMod = [ones(size(dataTest,1),1) dataTest];

alpha = 0.0001;
iter_max = 5000000;
threshold = 10^-3;

weight = zeros(1,features+1);

previousLikelihood = -Inf;

for i = 1 : iter_max
    
    currentLikelihood = likelihood(dataTrainMod,dataTargetTrain,weight);
    
    gradient = sum((repmat((dataTargetTrain- sigmoid(dataTrainMod,weight)),1,features+1).* dataTrainMod));
    
    weight = weight + alpha * gradient;
       
    if abs(currentLikelihood - previousLikelihood) < threshold
        break
    end
    previousLikelihood = currentLikelihood;
    fprintf('Iteration = %d --- likelihood = %0.2f\n', i, currentLikelihood);
    
end

classification = sigmoid(dataTestMod,weight) > 0.5;
accuracy = 100 - 100 * (1- sum(classification ~= dataTargetTest)/size(dataTargetTest,1))






