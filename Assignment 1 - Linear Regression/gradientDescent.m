function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
    % theta = zeros(2, 1); % initialize fitting parameters
    
    % h_theta(x_i) = theta_0 * x_0 + theta_1 * x_1  %% x_0 = 1
    
    tempTheta0 = 0;
    tempTheta1 = 0;
    
    for i =1:m
        tempTheta0 = tempTheta0 + X(i,1)*(theta(1,1)*X(i,1) + theta(2,1)*X(i,2) - y(i));
        tempTheta1 = tempTheta1 + X(i,2)*(theta(1,1)*X(i,1) + theta(2,1)*X(i,2) - y(i));
    end
    
    theta(1,1) = theta(1,1) - (alpha/m)*tempTheta0;
    theta(2,1) = theta(2,1) - (alpha/m)*tempTheta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
