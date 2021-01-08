% svm 시작
clear;
close all;
clc;

% 그래프
fprintf('Loading and Visualizing Data ...\n')
load('ex6data1.mat');
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% 리니어 + svm
load('ex6data1.mat');
fprintf('\nTraining Linear SVM ...\n')
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;



% 가우시안 커널 정의
fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1];
x2 = [0 4 -1];
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
% \t = 들여쓰기
fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);
fprintf('Program paused. Press enter to continue.\n');
pause;

% 그래프
fprintf('Loading and Visualizing Data ...\n')
load('ex6data2.mat');
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% RBF 커널 정의
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
load('ex6data2.mat');
C = 1;
sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;

% 그래프
fprintf('Loading and Visualizing Data ...\n')
load('ex6data3.mat');
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% rbf 커널 + svm
load('ex6data3.mat');
[C, sigma] = dataset3Params(X, y, Xval, yval);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
pause;