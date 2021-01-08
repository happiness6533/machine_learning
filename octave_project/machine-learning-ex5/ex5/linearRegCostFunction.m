function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  % 코스트함수 계산
  m = length(y);
  J = sum((((X * theta) - y).^2)) / (2 * m) ...
  + (lambda / (2 * m)) * sum(sum(theta(2:end, :).^2));

  % 미분계수 계산
  grad = (X' * ((X * theta) - y)) / m;
  grad(2:end, :) += (lambda / m) * theta(2:end, :);
  grad = grad(:);
end