% function [리턴값 여러개] = 함수
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters,
    theta = theta - alpha * (X' * (X * theta - y)) / m;
    J_history(iter) = computeCostMulti(X, y, theta);
  end
end