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
m = size(X, 1);

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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
ytemp = zeros(m, num_labels);
for i = 1:m
	ytemp(i, y(i)) = 1;
endfor
y = ytemp;
X = [ones(m, 1) X];
bigSum = 0;
gradSum2 = 0;
gradSum1 = 0;
for i = 1 : m
	xtemp = X(i, :);
	ytemp = y(i, :);
	ztwo = xtemp * Theta1';
	atwo = sigmoid(ztwo);
	atwo = [1 atwo];
	zthree = atwo * Theta2';
	athree = sigmoid(zthree);
	h = athree;
	temp2 = -ytemp .* log(h) - (1-ytemp) .* log(1 - h);
	bigSum = bigSum + sum(temp2);

	delta3 = h - ytemp;
	gprime2 = atwo .* (1 - atwo);

	delta2 = delta3 * Theta2 .* gprime2;
	gradSum2 = gradSum2 + delta3' * atwo;

	gprime1 = xtemp .* (1 - xtemp);
	delta1 = Theta1' * delta2(2:end)' .* gprime1;
	gradSum1 = gradSum1 + delta2(2:end)' * xtemp;

endfor
J = bigSum / m;

regTheta1 = Theta1;
regTheta1(:, 1) = 0;
regTemp1 = regTheta1 .* regTheta1;

regTheta2 = Theta2;
regTheta2(:, 1) = 0;
regTemp2 = regTheta2 .* regTheta2;

J = J + lambda * (sum(sum(regTemp1)) + sum(sum(regTemp2))) / (2 * m);

Theta1_grad = (gradSum1 + lambda * regTheta1) / m;
Theta2_grad = (gradSum2 + lambda * regTheta2) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
