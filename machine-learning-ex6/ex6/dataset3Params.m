function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.1;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================

#{
c_array = [0.01,0.03];
sigma_array = [0.01,0.03];

c_result = 0;
sigma_result = 0;
current_mean = zeros(2,2);

for i=1:columns(c_array)
	for j=1:columns(sigma_array)
		model= svmTrain(X, y, c_array(1,i), @(x1, x2) 			gaussianKernel(x1, x2, sigma_array(1,j)));
		predictions = svmPredict(model,Xval);
		current_mean(i,j) = mean(double(predictions ~= yval));			
	end
end

least_error = current_mean(1,1);
least_c = 1;
least_sigma = 1;
for i=1:rows(current_mean)
	for j=1:columns(current_mean)
		if(least_error>current_mean(i,j))
			least_error = current_mean(i,j);
			least_c = i;
			least_sigma=j;
		end
	end
end

current_mean
C = c_array(1,i)
sigma = sigma_array(1,j)
#}

C=1
sigma = 0.1
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
