function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h =  X * theta;
J =  sum((h - y).^2) / (2*m);
%grad = sum((h - y).*X)/(2*m);
%ttt = h-y;
grad = (1/m).*(X'*(h-y));
temp = theta;
temp(1) = 0;
grad = grad + (lambda/m).*temp;
J = J + (lambda/(2*m))*sum(temp.^2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
