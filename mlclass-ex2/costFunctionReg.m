function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = sigmoid(X*theta);
J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1 - h));
reg = 0;
for i = 2:size(theta,1)
    reg = reg + theta(i)^2;
end;
J = J + (lambda/(2*m))*reg;

grad = zeros(size(theta));
temp = zeros(size(theta));

temp(1) = (1/m)*sum((h - y).*X(:,1));

for j = 2:size(theta,1),
    %temp(j) = theta(j) - (alpha/m)*sum((prediction - y).*X(:,j))
    temp(j) = (1/m)*sum((h - y).*X(:,j)) + (lambda/m)*theta(j);
end;
for j = 1:size(theta,1),
    grad(j) = temp(j)
end;





% =============================================================

end
