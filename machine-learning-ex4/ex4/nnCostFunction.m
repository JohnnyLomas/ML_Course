function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % number of training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% ======Begin cost function computation==============================================

X = [ones(m, 1) X];              % Add column of ones to the X matrix

% ======Compute the hypothesis for each class for each training example==============

z1 = sigmoid(X*Theta1'); 		 % Compute the activations for the hidden layer
z1 = [ones(m, 1) z1];    		 % Add the bias unit to the hidden layer
h = sigmoid(z1*Theta2');  		 % Compute the output layer

% =====Convert y vector to matrix of ones and zeros (training examples x classes)====

y_mat = zeros(m, num_labels);    % Initialize y matrix 
for i = 1:m				 		 % Iterate through the y vector
		y_mat(i,y(i)) = 1;		 % Update y matrix
endfor

% =====Compute the cost function using for loops=====================================

for i = 1:m						 % Sum over training examples
	for k = 1:num_labels 		 % Sum over number of classes
								 % Compute the cost function
		J = J + ((y_mat(i,k))*log(h(i, k)) + (1 - y_mat(i, k))*log(1 - h(i, k)))/(-m);
								 
	endfor
endfor

% =====Add the regularization term to the cost=====================================

% Compute the regularization term (excluding bias units!)
reg_cost = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Add the regularization term to the unregularized cost
J = J + reg_cost;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% =====Convert y vector to matrix of ones and zeros (training examples x classes)====

y_mat = zeros(m, num_labels);    % Initialize y matrix 
for i = 1:m				 		 % Iterate through the y vector
		y_mat(i,y(i)) = 1;		 % Update y matrix
endfor

%======Compute gradients using backpropagation=======================================

for t = 1:m        	% Iterate over all training examples
	
	% Step 1: Perform a feedforward pass with a training example,
	%         computing activations for hidden layer and output layer
	
	   a_1 = X(t, :);
	   a_2 = sigmoid(a_1*Theta1');      % Compute activations of hidden layer
	   a_2 = [1 a_2]; 			        % Add the bias unit to the hidden layer
	   a_3 = sigmoid((a_2*Theta2'))'; 	% Compute outputs in layer 3
	
	% Step 2: Set the output layer error terms
	
	   delta_3 = a_3 - y_mat(t, :)';
	
	% Step 3: Set the hidden layer error terms
	
	   delta_2 = (Theta2'*delta_3).*[1; sigmoidGradient(a_1*Theta1')'];
	   
	% Step 4: Accumulate gradients while skipping the bias unit in layer 2
	
	   Theta1_grad = Theta1_grad + delta_2(2:end)*X(t, :);
	   Theta2_grad = Theta2_grad + delta_3*a_2;
	   
endfor

	% Step 5: Divide the gradients by the number of training examples
	   
	   Theta1_grad = Theta1_grad/m;
	   Theta2_grad = Theta2_grad/m;
	

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Regularize Theta1_grad matrix
size_T1 = size(Theta1);
for i = 1:size_T1(1)
	for j = 2:size_T1(2)
		Theta1_grad(i, j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i, j);
	endfor
endfor

% Regularize Theta2_grad matrix
size_T2 = size(Theta2);
for i = 1:size_T2(1)
	for j = 2:size_T2(2)
		Theta2_grad(i, j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i, j);
	endfor
endfor


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
