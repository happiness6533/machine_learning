function [U, S] = pca(X)

% ���л�(�ñ׸�)�� ���̰պ��� + ���̰պ��Ͱ� ���ϱ�
m = size(X, 1);
n = size(X, 2);
sigma = (X' * X) / m;
[U, S, V] = svd(sigma);

end