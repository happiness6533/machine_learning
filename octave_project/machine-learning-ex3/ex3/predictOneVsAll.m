function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  X = [ones(m, 1) X];
  p = zeros(m, 1);

  [temp, p] = max(sigmoid(X * all_theta'), [], 2);
end