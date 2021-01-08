% function 리턴값 1개 = 함수
function J = computeCost(X, y, theta)
  m = length(y);
  J = ((X * theta - y)' * (X * theta - y)) / (2 * m);
end