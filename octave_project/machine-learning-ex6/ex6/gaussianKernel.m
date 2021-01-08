function sim = gaussianKernel(x1, x2, sigma)
  x1 = x1(:);
  x2 = x2(:);
  % 두 점(벡터)간의 비슷한 정도를 0 - 1 사이의 값으로 정의한다
  sim = exp(-sum((x1 - x2).^2) / (2 * sigma.^2));
end