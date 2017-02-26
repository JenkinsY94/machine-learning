function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% do not count bias weight when computing cost J.
h = X * theta;
diff = h - y;
J = 1/(2*m) * sumsqr(diff) + lambda/(2*m) * sumsqr(theta(2:end));


num = size(grad, 1);
for i=1:num
    if i == 1
        grad(i) = 1/m * X(:,i)' * diff;
    else
        grad(i) = 1/m * X(:,i)' * diff + lambda/m * theta(i);
    end
end
% =========================================================================

grad = grad(:);

end
