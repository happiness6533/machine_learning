clear;
close all;
clc;

% ������ �ε�
fprintf('Visualizing example dataset for outlier detection.\n\n');
load('ex8data1.mat');
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
fprintf('Program paused. Press enter to continue.\n');
pause

% ��հ� �л� ���ϱ�
fprintf('Visualizing Gaussian fit.\n\n');
[mu sigma2] = estimateGaussian(X);

% ����þ� ���� pdf ���ϱ�
p = multivariateGaussian(X, mu, sigma2);

% ��հ� �л� �ð�ȭ
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
fprintf('Program paused. Press enter to continue.\n');
pause;

% �򰡼����� ���� �׽�Ʈ
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

outliers = find(p < epsilon);
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
fprintf('Program paused. Press enter to continue.\n');
pause;


% ������ �ε�
load('ex8data2.mat');

%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));
