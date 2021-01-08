function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

  for i = 1:length(lambda_vec)
    lambda = lambda_vec(i, :);
    theta = trainLinearReg(X, y, lambda);
      
    J = linearRegCostFunction(X, y, theta, 0);
    error_train(i, :) = J;
    
    J = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i, :) = J;
  endfor
end