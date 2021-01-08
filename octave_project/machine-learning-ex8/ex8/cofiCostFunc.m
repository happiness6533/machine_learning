function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
                                  
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

% �ڽ�Ʈ �Լ��� ����� �ƴ϶� ���̹Ƿ�, sum�� ���Ѵ�
J = sum(sum(((X * Theta').* R - Y).^2)) / 2 ...
    + (lambda / 2) * sum(sum(X.^2)) ...
    + (lambda / 2) * sum(sum(Theta.^2));
    
X_grad = ((X * Theta').* R - Y) * Theta + lambda * X;
Theta_grad = ((X * Theta').* R - Y)' * X + lambda * Theta;
grad = [X_grad(:); Theta_grad(:)];

end