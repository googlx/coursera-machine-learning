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

% 第一部分：求代价函数
%     参数：1. 将y转化为0/1矩阵；2. 利用前匮算法计算h值
%     前匮算法涉及多步矩阵运算，最好预先画出矩阵及其运算步骤，而后再编程
A1 = [ones(m, 1) X]';
Z2 = Theta1 * A1;
A2 = [ones(1, m); sigmoid(Z2)];
Z3 = Theta2 * A2;
H = sigmoid(Z3)';
Y = zeros(m, num_labels);
for a=1:m,
	Y(a, y(a)) = 1;
end;
J = -(1/m) * sum(sum((Y.*log(H)+(1-Y).*log(1-H))));

T1 = Theta1(:, 2:end); 
T2 = Theta2(:, 2:end); % 去除第一个Theta值
eye1 = eye(size(T1, 1));
eye2 = eye(size(T2, 1)); % 辅助计算对角线元素
reg = lambda / (2*m) * (sum(sum((T1*T1' .* eye1))) + sum(sum(T2*T2' .* eye2))); % 正规化代价函数，注意只累加对角线上的元素
J += reg;

% 第二部分：实现反向传播算法，计算Theta的梯度
%     参数：1. H矩阵，Y矩阵；2. Theta矩阵和A矩阵
Delta3 = H - Y;
Delta2 = T
for a=1:m,
	Delta1 = 

% 第三部分：正则化代价函数和梯度
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
