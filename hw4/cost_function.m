clear all;
close all;
clc;

%% load data
load('data2.mat');
K = max(labels);
C = zeros([K, 2]);
X = X';
labels = labels';
for i = 1:K
	C(i,:) = sum(X(labels==i, :))/sum(labels==i);
endfor
[J] = cost_func(X', labels', C, K);	


function [J] = cost_func(X, Y, C, k)
	J = 0;
	for i = 1:k
		dif = X(Y==i, :) - repmat(C(i, :), sum(Y==i), 1);
		J += sum(sum(dif .^ 2, 2));
	endfor	
endfunction
