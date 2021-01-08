function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  % �ڽ�Ʈ�Լ� ���
  m = length(y);
  J = sum((((X * theta) - y).^2)) / (2 * m) ...
  + (lambda / (2 * m)) * sum(sum(theta(2:end, :).^2));

  % �̺а�� ���
  grad = (X' * ((X * theta) - y)) / m;
  grad(2:end, :) += (lambda / m) * theta(2:end, :);
  grad = grad(:);
end