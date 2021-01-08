function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = size(X, 1);
  n = size(X, 2);
  
  J = -((y' * log(sigmoid(X * theta))) ...
  + (ones(m, 1) - y)' * (log(ones(m, 1) - sigmoid(X * theta)))) / m ...
  + lambda * theta(2:n, :)' * theta(2:n, :) / (2 * m);

  grad = (X' * (sigmoid(X * theta) - y)) / m ...
  + lambda * theta / m;
  grad(1, :) = X(:, 1)' * (sigmoid(X * theta) - y) / m;
end