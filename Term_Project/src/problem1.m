%% ********************************************************************************* %% 
clc;clearvars

data = xlsread('pima_indians_diabetes.csv');

%% Selecting only 100 data points and targets
glucoseConc = data(1:100,2);
target = data(1:100,9);

%% Calulating the prior estimated form MLE
prior1 = sum(target(:) == 1)/size(glucoseConc,1);
prior2 = sum(target(:) == 0)/size(glucoseConc,1);

dispPriors = ['The estimated priors p1 and p2 are ',num2str(prior1),' & ',num2str(prior2)];
disp(dispPriors)

%% Calculating the means and variances of the two likelihoods
mean1 = sum(glucoseConc(target(:) == 1))/size(glucoseConc(target(:) == 1),1);
mean2 = sum(glucoseConc(target(:) == 0))/size(glucoseConc(target(:) == 0),1);

dispMeans = ['The estimated means m1 and m2 are ',num2str(mean1),' & ',num2str(mean2)];
disp(dispMeans)

var1 = sum((glucoseConc(target(:) == 1) - mean1).^2)/size(glucoseConc(target(:) == 1),1);
var2 = sum((glucoseConc(target(:) == 0) - mean2).^2)/size(glucoseConc(target(:) == 0),1);

dispVars = ['The estimated variances v1 and v2 are ',num2str(var1),' & ',num2str(var2)];
disp(dispVars)

%% Validation

glucoseConcVal = data(101:end,2);
targetVal = data(101:end,9);

evidence = normpdf(glucoseConcVal,mean1,sqrt(var1)) * prior1 + normpdf(glucoseConcVal,mean2,sqrt(var2)) * prior2;

posteriorClass1 = exp((log(normpdf(glucoseConcVal,mean1,sqrt(var1))) + log(prior1)) - log(evidence));
posteriorClass2 = exp((log(normpdf(glucoseConcVal,mean2,sqrt(var2))) + log(prior2)) - log(evidence));

%posteriorClass1 = (normpdf(glucoseConcVal,mean1,sqrt(var1) .* prior1))./evidence;
%posteriorClass2 = (normpdf(glucoseConcVal,mean2,sqrt(var2) .* prior2))./evidence;

%a = aposteriorClass1 - posteriorClass1;
%b = aposteriorClass2 - posteriorClass2;

classification = posteriorClass1 - posteriorClass2 > 0;

accuracy = (size(targetVal,1) - sum(abs(classification - targetVal)))/size(targetVal,1);

disp(['The accuracy obtained is ',num2str(accuracy)])

%% ******************************************************************************** %%


