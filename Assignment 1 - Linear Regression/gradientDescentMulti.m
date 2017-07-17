function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    tempTheta0 = 0;
    tempTheta1 = 0;
    tempTheta2 = 0;
    
    for i = 1:m
      tempTheta0 = tempTheta0 + X(i,1)*(theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3) - y(i));
      tempTheta1 = tempTheta1 + X(i,2)*(theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3) - y(i));
      tempTheta2 = tempTheta2 + X(i,3)*(theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3) - y(i));
    end
    
    theta(1,1) = theta(1,1) - (alpha/m)*tempTheta0;
    theta(2,1) = theta(2,1) - (alpha/m)*tempTheta1;
    theta(3,1) = theta(3,1) - (alpha/m)*tempTheta2;
    
    %% APPARENTLY IT WORKS BUT SINCE IT'S NOT GENERALIZED (how many features you want),
    %% THEN THE COURSE IS NOT ACCEPTING IT. I should also vectorize it.

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
