function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = sigmoid(X*theta);
J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1 - h));
grad = zeros(size(theta));

%grad = (1/m)*sum((h - y) * X);

for j = 1:size(theta,1),
    %temp(j) = theta(j) - (alpha/m)*sum((prediction - y).*X(:,j))
    temp(j) = (1/m)*sum((h - y).*X(:,j))
end;
for j = 1:size(theta,1),
    grad(j) = temp(j)
end;








% =============================================================

end
