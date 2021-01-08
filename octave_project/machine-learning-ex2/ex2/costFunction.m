function [J, grad] = costFunction(theta, X, y)
  m = length(y);

  J = -((y' * log(sigmoid(X * theta))) 
  + ((ones(m, 1) - y)' * log(ones(m, 1) - sigmoid(X * theta)))) / m;
  
  grad = (X' * (sigmoid(X * theta) - y)) / m;
end