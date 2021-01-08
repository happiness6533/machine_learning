function p = predict(theta, X)
  % X�� 1�� ���Ͱ� �ƴϱ� ������, �Ʒ��� ���� ������� ���� ������ �����Ѵ�
  m = size(X, 1);
  p = zeros(m, 1);
  
  for i = 1:m,
    if (X * theta)(i, :) > 0,
      p(i, :) = 1;
    end
end