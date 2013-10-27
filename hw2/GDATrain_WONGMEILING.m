function [phi, mu0, mu1, Sigma] = GDATrain_WONGMEILING( DataTrain, LabelsTrain )

% convert the labels to the range {0,1}
LabelsTrain = LabelsTrain>0;

% initialize dimension (number of training data)
dim = size(DataTrain,2);
m = size(DataTrain,1);

%sum of where Label equals 1
posLabels = sum(LabelsTrain);

%sum of where Label equals 0
negLabels = m - posLabels;

%compute phi 
phi = posLabels/m;

%compute mean vector mu0 - select the row of the training data where Label = 0
mu0 = sum(DataTrain(~LabelsTrain, : ))/negLabels;
mu0 = mu0';

%compute mean vector mu1 - select the row of the training data where Label = 1
mu1 = sum(DataTrain(LabelsTrain, : ))/posLabels;
mu1 = mu1';

%compute Sigma (covariance)
	
	%first compute the mean-matrix (of mu0 and mu1) - the i-th row contains mu0 if the i-th Label = 0
	%(mu1 if the i-th Label = 1)
	%Example: Label = [0 0 1], mu0 = 2, mu1 = 3 ==> muMatrix =     2 2 2... 2
	%					  2 2 2... 2
	%					 3 3 3... 3
						
	muMatrix = (repmat(mu0, 1, m)' .* repmat(~LabelsTrain, dim,1)') + (repmat(mu1, 1, m)' .* repmat(LabelsTrain, dim,1)');
	
	Sigma = ((DataTrain - muMatrix)' * (DataTrain - muMatrix))/m;

end