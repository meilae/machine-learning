function [C, A] = Kmeans_WONGMEILING(X, Cinit)

% dummy
%C = Cinit;
%A = ones(1,size(X,2)); % this is not a proper assignment

% number of clusters
k = size(Cinit, 2);

% initial labels
Y = zeros(size(X, 2), 1);

[C, Y] = kmeans_rec(Cinit', X', Y, k);

A = Y';
C = C';

end

% A recursive function to compute k-means clustering
function [C, Y] = kmeans_rec(Cinit, X, prevY, k)
        C = zeros([k, size(Cinit, 2)]);
        
	%returns the indices y of the closest points in
	%Cinit for each point in X. 
	%Cinit is an m-by-n matrix representing m points 
	%in n-dimensional space. X is a p-by-n matrix, 
	%representing p points in n-dimensional space. 
	%The output y is a column vector of length p.
	Y = dsearchn(Cinit, X); 
	
	% compute the new cluster centroids (mean of the clustering points)	
        for i = 1:k 
                C(i,:) = sum(X(Y==i, :))/sum(Y==i); %count the number of occurrences of i
        endfor

	% compare the labels of the previous clustering with the current ones. 
	%If they are not equal, so run the recursive k-means clustering function again.
        if ~(isequal(Y, prevY))
                [C, Y] = kmeans_rec(C, X, Y, k);
        end
endfunction


