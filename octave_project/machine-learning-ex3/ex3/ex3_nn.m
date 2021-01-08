% 뉴럴 네트워크 propagation
clear;
close all;
clc;


% 데이터 로드
% 이미지 = 20 * 20 = 400개의 픽셀 데이터
% 이미지 데이터를 0 - 9 분류
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10; 

fprintf('Loading and Visualizing Data ...\n')
load('ex3data1.mat');
m = size(X, 1);


% 랜덤 데이터 100개 디스플레이
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
fprintf('Program paused. Press enter to continue.\n');
pause;


% 세타 로드
fprintf('\nLoading Saved Neural Network Parameters ...\n')
load('ex3weights.mat');


% 정확도 출력
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;


% 예제 이미지 랜덤으로 출력 + 정확도 출력
rp = randperm(m);
for i = 1:m
  fprintf('\nDisplaying Example Image\n');
  displayData(X(rp(i), :));

  pred = predict(Theta1, Theta2, X(rp(i),:));
  fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
  % 원한다면 끝
  s = input('Paused - press enter to continue, q to exit:','s');
  if s == 'q'
    break
  end
end