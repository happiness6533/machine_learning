function [X_poly] = polyFeatures(X, p)
  X_poly = X;
  for i = 2:p,
    X_poly = [X_poly, X.^i];
  endfor
end