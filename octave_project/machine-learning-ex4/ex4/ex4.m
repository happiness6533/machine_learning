% 뉴럴 네트워크
clear;
close all;
clc;


% 데이터 로드
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10;
fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');
m = size(X, 1);


% 랜덤 데이터 100개 디스플레이
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
pause;


% 세타 로드 및 코스트 계산
fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('ex4weights.mat'); 
nn_params = [Theta1(:) ; Theta2(:)];
fprintf('\nFeedforward Using Neural Network ...\n')
lambda = 0;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% 오버피팅 해결
fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;


% 시그모이드그래디언트 함수
fprintf('\nEvaluating sigmoid gradient...\n')
g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% 파라미터 초기화
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


% 파라미터 벡터화
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Backpropagation
fprintf('\nChecking Backpropagation... \n');
checkNNGradients;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')


% Backpropagation + 오버피팅 해결
lambda = 3;
checkNNGradients(lambda);
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);
fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);
fprintf('Program paused. Press enter to continue.\n');
pause;


% 최적화
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 0.1;

% p를 변수로 하는 함수 costFunction을 생성한다
% p = 파라미터 벡터
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% 세타 reshape
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.\n');
pause;


% 히든 레이어 시각화
fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% 정확도 출력
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);