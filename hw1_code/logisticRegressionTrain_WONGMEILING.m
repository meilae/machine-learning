function [ theta ] = logisticRegressionTrain_WONGMEILING( DataTrain, LabelsTrain, maxIterations )
% logisticRegressionTrain train a logistic regression classifier
% [ theta ] = logisticRegressionTrain( DataTrain, LabelsTrain, MaxIterations )
% Using the training data in DataTrain and LabelsTrain trains a logistic
% regression classifier theta. 
% 
% Implement a Newton-Raphson algorithm.

% convert the labels to the range {0,1}
LabelsTrain = (LabelsTrain + 1) ./ 2;

% function to compute h_theta(x)
sigmoid = @(x) 1 ./ (1 + exp(-x));

% initialize dimension, number of training data
dim = size(DataTrain,2);
theta = zeros(dim,1);

m = size(DataTrain,1);

% maximization
for i=1:maxIterations

	% compute the gradient of log likelihood
	gradient_log_likelihood = zeros(1,dim);
	
	for k=1:m
		gradient_log_likelihood = gradient_log_likelihood + (LabelsTrain(k) - sigmoid(theta' * DataTrain(k,:)')) * DataTrain(k,:);
	end
	
	gradient_log_likelihood = gradient_log_likelihood / m;

	% compute the hessian of log likelihood
	hessian_log_likelihood = zeros(dim);

	for l=1:m
		hessian_log_likelihood = hessian_log_likelihood + sigmoid(theta' * DataTrain(l,:)' ) .* (1 - sigmoid(theta' * DataTrain(l,:)' )) .* DataTrain(l,:)' * DataTrain(l,:);
	end

	hessian_log_likelihood = -hessian_log_likelihood ./ m;

	% update theta
	theta = theta - inv(hessian_log_likelihood) * gradient_log_likelihood';

end
