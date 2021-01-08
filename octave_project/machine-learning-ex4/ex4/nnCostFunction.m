function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

  % 세타 재생성
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
  
  % 세타 세로화 (= 입출력 가로화)
  Theta1 = Theta1';
  Theta2 = Theta2';  
                 
  % 레이어 생성
  m = size(X, 1);

  pre_hidden = [ones(m, 1), X] * Theta1;
  hidden = sigmoid(pre_hidden);
  
  pre_output = [ones(m, 1), hidden] * Theta2;
  output = sigmoid(pre_output);

  % y 재설정
  temp1 = zeros(size(output));
  temp2 = [1:1:size(output, 2)];
  for i = 1:m,
    temp1(i, :) = (temp2 == y(i, :));
  end
  y = temp1;
    
  % 코스트 함수
  J = -sum(sum(y .* log(output) ... 
  + (ones(size(output)) - y) .* log(ones(size(output)) - output))) / m ...
  + lambda * sum(sum((Theta1(2:end, :)).^2)) / (2 * m) ...
  + lambda * sum(sum((Theta2(2:end, :)).^2)) / (2 * m);
  
  % 미분계수
  Theta2_pre_grad = output - y;
  Theta2_grad = ([ones(m, 1), hidden]' * Theta2_pre_grad) / m;

  Theta1_pre_grad = (Theta2_pre_grad * Theta2(2:end, :)') ...
  .* sigmoidGradient(pre_hidden);
  Theta1_grad = ([ones(m, 1), X]' * Theta1_pre_grad) / m;

  Theta2_grad(2:end, :) += (lambda / m) * Theta2(2:end, :);
  Theta1_grad(2:end, :) += (lambda / m) * Theta1(2:end, :);

  grad = [Theta1_grad'(:); Theta2_grad'(:)];
end