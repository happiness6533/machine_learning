function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1); % За
  X = [ones(m, 1), X];
  n = size(X, 2); % ї­
  
  all_theta = zeros(num_labels, n);

  initial_theta = zeros(n, 1);

  for c = 1:num_labels,
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    initial_theta, options);
    all_theta(c, :) = theta';
  end
end