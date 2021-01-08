% k-means
clear;
close all;
clc;


% �� �����͸� ���� ����� ���Ϳ� ����
fprintf('Finding closest centroids.\n\n');
load('ex7data2.mat');
K = 3;
initial_centroids = [3 3; 6 2; 8 5];
idx = findClosestCentroids(X, initial_centroids);
fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% �� ���Ϳ� ����Ǿ� �ִ� �����͵��� ��� ���ϱ�
% ��� = ���ο� ����
fprintf('\nComputing centroids means.\n\n');
centroids = computeCentroids(X, idx, K);
fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% k-means ����
fprintf('\nRunning K-Means clustering on example dataset.\n\n');
load('ex7data2.mat');
K = 3;
max_iters = 10;
initial_centroids = [3 3; 6 2; 8 5];
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% k-means �̹��� ����
% 256���� ����� �̷���� �̹����� �ȼ��� ������ ������ȭ
% ���� 16�� ���� / Ŭ�����͸�
% ���� ���Ϳ� �ش��ϴ� �ȼ� �����ʹ� ���� ��(���� ��)����!
% 16���� ����(���͵�)�θ� �̷���� �̹��� ���� / ���� ����
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');
load('bird_small.mat');

% �ȼ��� ��� ���� 0 - 1 ���̿� ������ 255�� ������
A = A / 255;
img_size = size(A);

% A = �ȼ� �� * �ȼ� �� * r/g/b >> A = �ȼ����� * r/g/b �� reshape
X = reshape(A, img_size(1) * img_size(2), 3);

K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
fprintf('Program paused. Press enter to continue.\n');
pause;

printf('\nApplying K-Means to compress an image.\n\n');
idx = findClosestCentroids(X, centroids);

% a(b, :) = a��� �߿��� b����� �� ���Ұ��� �ش��ϴ� ���� ���ʴ�� ����Ѵ�
% a(b, :) = ���� ���� ����Ѵ�
% ���ʴ�� = b����� ������ �Ʒ��� �����ϸ鼭 1�� Ŭ���� >> 2�� Ŭ���� >> ...
X_recovered = centroids(idx, :);

% ������ ���� �̹��� 3���� ��ķ� �����
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% �������� �̹��� ���
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% ������ ���� �̹��� ���
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
fprintf('Program paused. Press enter to continue.\n');
pause;