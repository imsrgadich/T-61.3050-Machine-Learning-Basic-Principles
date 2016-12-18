%% ********************************************************************************* %% 
%%Here class2 is (Class = 1) in problem and class1 is (Class = 0)%%

%% ********************************************************************************* %% 
clc;clearvars;

trainingData = load('spect_training.txt');
testData = load('spect_test.txt');

classes = 2;
features = 22;

%% Separating the training inputs and targets!
trainingInput = trainingData(:,2:end);
trainingTarget = trainingData(:,1);

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


