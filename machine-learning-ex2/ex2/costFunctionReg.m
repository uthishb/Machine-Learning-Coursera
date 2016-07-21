function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


[thetaX thetaY] = size(theta);
sum2=0;
theta
for iter = 2:thetaX
	sum2 = sum2+theta(iter,1).^2	;
end

sum2 = (lambda/(2*m)).*sum2;

sum1 = (1/m).*(sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta))));

J = sum1 + sum2;

grad(1,1) = (1/m).*sum((sigmoid(X*theta)-y).*(X(:,1)));

[XRow XColumn] = size(X);
for iter = 2:XColumn
	grad(iter,1) = (1/m).*sum((sigmoid(X*theta)-y).*(X(:,iter)))+((lambda/m)*theta(iter,1));
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
