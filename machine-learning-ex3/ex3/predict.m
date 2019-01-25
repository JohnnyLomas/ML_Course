function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % number of training examples
num_labels = size(Theta2, 1); % number of classes 

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % Return predictions in a vector with the same number as training set

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add column of ones to the X matrix
X = [ones(m, 1) X];

% Compute the activations for the hidden layer
z1 = sigmoid(X*Theta1');

%% Add the bias unit to the hidden layer 
z1 = [ones(m, 1) z1];

% Compute the output layer
z2 = sigmoid(z1*Theta2');

% Select the maximum probability from each class and store it in a vector
[max, p] = max(z2, [], 2);

% Replace 10 with 0 due to indexing quirk
for i = 1:length(p)
	if p(i) == 10
		p(i) = 0;
	endif
endfor

% =========================================================================


end
