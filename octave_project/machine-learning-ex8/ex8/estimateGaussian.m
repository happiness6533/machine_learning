function [mu sigma2] = estimateGaussian(X)

m = size(X, 1);

% ���
mu = sum(X) ./ m;

% �л�
sigma2 = X;
for i = 1:m
  sigma2(i, :) = (X(i, :) - mu).^2;
endfor
sigma2 = sum(sigma2) ./ m;

end