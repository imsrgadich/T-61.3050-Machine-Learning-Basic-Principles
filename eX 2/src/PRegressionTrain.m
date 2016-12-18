function [weight,errorTest] = PRegressionTrain(degree)
clearvars -except degree

load training_data.txt

n_data = 50;
perm = randperm(n_data);
training_idx = perm([1:24 27:end]);
test_idx = perm(25:26);

trainingData = training_data(training_idx);
validationData = training_data(test_idx);
targets = training_data(training_idx,2);
outputVal = training_data(test_idx,2);

%% Build the phi (�) matrix-----------%%

power = 0:degree; %%Create a power array
power_train = repmat(power,48,1); %%Repeat the power array 
phi = repmat(trainingData',1,degree+1);
phi = phi.^power_train;

%% Find the weights by pseudoinverse technique

weight = pinv(phi' * phi) * phi' * targets;

%% Validation

power_val = repmat(power,2,1); %%Repeat the power array
phi_val = repmat(validationData',1,degree+1);
phi_val = phi_val.^power_val;

%% Find the output and the error

output = phi_val * weight;

errorTest = mean((outputVal - output).^2);

text = ['The validation mean squared error for degree ',num2str(degree), ' is ',num2str(errorTest)];

disp(text)

end