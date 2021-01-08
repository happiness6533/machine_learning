function idx = findClosestCentroids(X, centroids)

m = size(X, 1);
idx = zeros(m, 1);
K = size(centroids, 1);

% �����Ϳ� ���� ����� ���� ����
for i = 1:m
  chosseCentroidIndex = 1;
  minDistance = sum((X(i, :) - centroids(1, :)).^2);
  for j = 2:K
    if minDistance > sum((X(i, :) - centroids(j, :)).^2)
      minDistance = sum((X(i, :) - centroids(j, :)).^2);
      chosseCentroidIndex = j;
    endif
  endfor
  % ���� ����� ����
  idx(i, :) = chosseCentroidIndex;
endfor
 
end