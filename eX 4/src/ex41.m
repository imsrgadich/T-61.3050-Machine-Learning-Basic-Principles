dt = load('diabete_data.csv');

N = 100;

training_data = dt(1:N, 1:8);
training_labels = dt(1:N, 9);

validation_data = dt(N+1:end, 1:8);
validation_labels = dt(N+1:end, 9);

% These don't depend on feature values
p1 = sum(training_labels == 0) / N;
p2 = 1 - p1;

% a-c) Train models for each feature

cl1 = find(training_labels == 0);
cl2 = find(training_labels == 1);

mu1s = mean(training_data(cl1, :));
mu2s = mean(training_data(cl2, :));

% ML variance
% Remove mean
train_cl1_c = bsxfun(@minus, training_data(cl1, :), mu1s);
train_cl2_c = bsxfun(@minus, training_data(cl2, :), mu2s);

% Check that means are now indeed (almost) zero
assert(all(mean(train_cl1_c) < 1e-10))
assert(all(mean(train_cl2_c) < 1e-10))

% The ML estimate for variance
sigmaML1s = sqrt(sum(train_cl1_c.^2)/length(cl1));
sigmaML2s = sqrt(sum(train_cl2_c.^2)/length(cl2));

% Unbiased variance estimate (divides by (n-1) rather than n)
sigmaUB1s = std(training_data(cl1, :));
sigmaUB2s = std(training_data(cl2, :));

% Classification
predictionsML = classify(validation_data, p1, p2, mu1s, mu2s, sigmaML1s, sigmaML2s);
predictionsUB = classify(validation_data, p1, p2, mu1s, mu2s, sigmaUB1s, sigmaUB2s);

% Note that the predictions are exactly the same, so we need only one accuracy
sum(sum(abs(predictionsML -predictionsUB)))

% Calculate accuracies
accuracyML = 1 - sum(abs(predictionsML - repmat(validation_labels, [1 8])))/size(validation_data, 1)

    

