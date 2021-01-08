% function [리턴값 여러개] = 함수
function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

  mu = [mean(X_norm(1, :)), mean(X_norm(2, :))];
  sigma = [std(X_norm(1, :)), std(X_norm(2, :))]; 
end