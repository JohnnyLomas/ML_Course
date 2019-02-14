%============================= FindParams ===================================
% 
% FindParams.m is a script designed to find the optimal values of C and 
% sigma to use when learning the data provided in ex6data3.m. It learns 64
% different models with different combinations of C and sigma, predicts the
% cross-validation set, and stores the error. It finishes by finding the 
% parameters associated with the smallest error.

load('ex6data3.mat');				% Load the data

C = [.01;.03;.1;.3;1;3;10;30];      % Provide the values of C and sigma 
sigma = C;

Error_matrix = zeros(64, 3);		% Empty matrix for storage

K = 1;
for i = 1:length(C)					% Iterate through C
	for j = 1:length(sigma)			% Iterate through sigma
		
		C_ = C(i);					% Get the value of C
		sigma_ = sigma(j);			% Get the value of sigma
		
		model = svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_));
									% Train the model on the training set
								
		predictions = svmPredict(model, Xval);
									% Generate predictions on the validation set
		
		error = mean(double(predictions ~= yval));
									% Compute the cross-validation error
		
		Error_matrix(K, 1) = error;	% Store values to find minimum at the end
		Error_matrix(K, 2) = C_;
		Error_matrix(K, 3) = sigma_;
		
		K = K +1;
		
	endfor
endfor

[minval, row] = min(min(Error_matrix(:, 1), [], 2));
									% Get the row index of the minimum error
minval
row
disp('C: ')							% Print the optimum values of C and sigma
Error_matrix(row, 2)
disp('sigma: ')
Error_matrix(row, 3)
Error_matrix
