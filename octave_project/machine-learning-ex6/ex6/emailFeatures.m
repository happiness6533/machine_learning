function x = emailFeatures(word_indices)

% ������ �ܾ� ����
n = 1899;

% �̸��� ����ȭ
x = zeros(n, 1);
wordNumber = size(word_indices, 1);
for i = 1:wordNumber,
  x(word_indices(i, :), :) = 1;
endfor

end