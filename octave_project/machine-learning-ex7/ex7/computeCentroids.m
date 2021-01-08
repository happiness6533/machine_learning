function centroids = computeCentroids(X, idx, K)

m = size(X, 1);
n = size(X, 2);

% ���� / ����ī��Ʈ ����
centroids = zeros(K, n);
centroidsCount = zeros(K, 1);

% ���� ���͸� ������ �����͵��� ��
for i = 1:K
  for j = 1:m
    if idx(j, :) == i
      centroids(i, :) += X(j, :);
      centroidsCount(i, :) += 1;
    endif
  endfor
endfor

% �� / ī��Ʈ = ���
centroids = centroids ./ centroidsCount;

end