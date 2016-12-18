data = load('pima_indians_diabetes.csv');
[m, n] = size(data);
trainingX = data(1:end, 1:8);
trainingX = (trainingX - mean(trainingX(:)))./ mean(trainingX(:));
trainingX = [ones(size(data,1), 1) trainingX];
trainingY = data(1:end, 9);

step_size = 0.001;
epoches_max = 3500;
fun_thres =  1e-5;

w = zeros(9,1);
lik_previous = -Inf;

epoch = 0;

fprintf('Using Gradient descent\n')
while(true)   
    	epoch = epoch + 1;
		y = 1 ./ (1 + exp(-trainingX * w));
		lik = sum(trainingY .* log(y) + (1 - trainingY) .* log(1 - y)); % given function in question
		grad = transpose(sum(bsxfun(@times, trainingX, (trainingY - y))));
        w = w  + (step_size * grad); 
        g(:,epoch) =grad;
        if (epoch >= epoches_max) || abs(lik - lik_previous) < fun_thres 
            break
        end
        lik_previous = lik;
        fprintf('epoch = %d --- liklihood = %0.2f\n', epoch, lik);
        
end

label_prediction = 1./(1+exp(-trainingX*w)) > 0.5;
100 * (1- sum(label_prediction ~= trainingY)/m)




