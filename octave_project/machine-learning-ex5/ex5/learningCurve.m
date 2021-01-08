function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
  m = size(X, 1);
  for i = 1:m
    theta = trainLinearReg(X(1:i, :), y(1:i, :), lambda);
   
    J = linearRegCostFunction(X(1:i, :), y(1:i, :), theta, 0);
    error_train(i, :) = J;
    
    J = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i, :) = J;
  end
end