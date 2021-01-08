function g = sigmoidGradient(z)
  g = sigmoid(z) .* (ones(size(z)) - sigmoid(z)); 
end