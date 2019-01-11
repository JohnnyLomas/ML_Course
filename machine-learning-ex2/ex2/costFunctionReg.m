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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ====================== Cost Function ============================
% Genrate the polynomial argument for the sigmoid function using matrix multiplication
z = X*theta;

% Compute the predicted value of y using the polynomial and the sigmoid function
h = sigmoid(z);

% Compute the total regularized cost function (vectorized)
J = (1/m)* ((-y')*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);

% ======================= Gradient ==============================
% Compute the gradient (vectorized)
grad = (1/m)*X'*(sigmoid(X*theta) - y);

% Add the regularization terms
reg = (lambda/m)*theta;
reg(1) = 0;
grad = grad + reg;


% =============================================================

end
