clc;clearvars;


data = xlsread('pima_indians_diabetes.csv');
[m, n] = size(data);
dataInput = data(1:100,1:8); 
dataTest= data(101:end,1:8);
dataTarget = data(1:100,9);
dataTargetTest = data(101:end,9);
data_new = (dataInput - mean(dataInput(:)))./ mean(dataInput(:));
dataTest = (dataTest - mean(dataTest(:)))./ mean(dataTest(:));
data_new = [ones(size(dataInput,1),1) data_new];
dataTest = [ones(size(dataTest,1),1) dataTest];

alpha = 0.001;
iter_max = 5000000;
threshold = 10^-6;

weight = zeros(1,9);

previousLikelihood = -Inf;

for i = 1 : iter_max
    
    currentLikelihood = likelihood(data_new,dataTarget,weight);
    
    gradient = sum((repmat((dataTarget - sigmoid(data_new,weight)),1,9).* data_new));
    
    weight = weight + alpha * gradient;
       
    if abs(currentLikelihood - previousLikelihood) < threshold
        break
    end
    previousLikelihood = currentLikelihood;
    fprintf('Iteration = %d --- likelihood = %0.2f\n', i, currentLikelihood);
    
end

classification = sigmoid(data_new,weight) > 0.5;
accuracy = 100 - 100 * (1- sum(classification ~= dataTarget)/m)

classification_test = sigmoid(dataTest,weight) > 0.5;
accuracy = 100 - 100 * (1- sum(classification_test ~= dataTargetTest)/m)




