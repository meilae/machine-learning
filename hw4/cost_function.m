clear all;
close all;
clc;

%% load data
load('data2.mat');
K = max(labels);
C = zeros([K, 2]);
X = X';
labels = labels';
J = 0;

for i = 1:K
	C(i,:) = sum(X(labels==i, :))/sum(labels==i);
endfor	

for i = 1:K
	dif = X(labels==i, :) - repmat(C(i, :), sum(labels==i), 1);
	J += sum(sum(dif .^ 2, 2));
endfor	

fprintf('the result of the cost function for the ground truth is %d\n', J);


