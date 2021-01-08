% ������ �ε�1
% Y = ���� * ����
% R = �������� = 1/0
fprintf('Loading movie ratings dataset.\n\n');
load ('ex8_movies.mat');
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));
imagesc(Y);
ylabel('Movies');
xlabel('Users');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ������ �ε�2
% params = X + Theta
load ('ex8_movieParams.mat');
num_users = 4;
num_movies = 5;
num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

% �ڽ�Ʈ �Լ� + �׷����Ʈ ��Ʈ(�������� x)
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 0);         
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Gradients (without regularization) ... \n');
checkCostFunction;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% �ڽ�Ʈ �Լ� + �׷����Ʈ ��Ʈ(�������� o)
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);          
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Gradients (with regularization) ... \n');
checkCostFunction(1.5);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ���� ��ȣ�ϴ� ��ȭ �� ���� ������ ����
movieList = loadMovieList();
my_ratings = zeros(1682, 1);
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;
fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ������ ������ + ���� ������
fprintf('\nTraining collaborative filtering...\n');
load('ex8_movies.mat');
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

% �Ķ���� ����
[Ynorm, Ymean] = normalizeRatings(Y, R);
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% ����ȭ �Լ��� ����ȭ�� X + Theta ���ϱ�
options = optimset('GradObj', 'on', 'MaxIter', 100);
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features, lambda)), ...
                initial_parameters, options);

X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);
fprintf('Recommender system learning completed.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ����ȭ�� X + Theta�� ���� �������� ��ȭ�� ���� ä���ֱ�
p = X * Theta';
my_predictions = p(:,1) + Ymean;
movieList = loadMovieList();
[r, ix] = sort(my_predictions, 'descend');

fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end