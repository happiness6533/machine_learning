function centroids = kMeansInitCentroids(X, K)

% m���� ������ X���� �����ϰ� ������ K���� �����͸� ���ͷ� �ʱ�ȭ
% randperm(10) : 1 - 10���� �����ϰ� ���ĵ� 1 * 10 ����� �����Ѵ�
randidx = randperm(size(X, 1));
centroids = X(randidx(1:K), :);

end