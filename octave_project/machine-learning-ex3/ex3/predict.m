function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  X = [ones(m, 1), X];
  
  [temp, index] = max([ones(m, 1), sigmoid(X * Theta1')] * Theta2', [], 2);
  p = index;
end