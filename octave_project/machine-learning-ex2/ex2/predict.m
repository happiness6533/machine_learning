function p = predict(theta, X)
  % X가 1열 벡터가 아니기 때문에, 아래와 같은 방법으로 행을 변수에 지정한다
  m = size(X, 1);
  p = zeros(m, 1);
  
  for i = 1:m,
    if (X * theta)(i, :) > 0,
      p(i, :) = 1;
    end
end