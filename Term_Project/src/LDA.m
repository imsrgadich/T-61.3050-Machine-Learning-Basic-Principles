clc,clearvars;

%% Red - 1 and White - 01
data = load('training_classification_regression_2015.csv');
labels = data(:,13);
regression_output = data(:,12);
data = data(:,1:11);

[m, n] = size(data);

%% mean normalization : comment it if you dont need it

mean = sum(data)/m;

data = data - repmat(mean,m,1);

data = data./(repmat(max(data),m,1));

%% LDA

%% Find the means of the two data i.e., is Red and White data
dataClass1 =data(labels==1,:);
dataClass0 =data(labels==0,:);
[m1,n1] = size(dataClass1);
[m0,n0] = size(dataClass0);

mean1 = sum(dataClass1)./m;
mean0 = sum(dataClass0)./m;

zero = dataClass0 - repmat(mean0,m0,1);
first = dataClass1 - repmat(mean1,m1,1);


%% Find covariance estimate

covariance = ((zero' * zero)./(m0)) + ((first' * first)./(m1));

w_opt = inv(covariance)*(mean1 - mean0)';

mod_data = data * w_opt;

figure
scatter(mod_data(labels==1),zeros(size(mod_data(labels==1),1),1),'o')
hold on
scatter(mod_data(labels==0),zeros(size(mod_data(labels==0),1),1),'o')
hold off

