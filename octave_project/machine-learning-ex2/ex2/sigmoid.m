function g = sigmoid(z)
  m = length(z);
  g = 1./(ones(m, 1) + exp(-z));
end