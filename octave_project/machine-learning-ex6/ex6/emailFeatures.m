function x = emailFeatures(word_indices)

% 사전의 단어 개수
n = 1899;

% 이메일 벡터화
x = zeros(n, 1);
wordNumber = size(word_indices, 1);
for i = 1:wordNumber,
  x(word_indices(i, :), :) = 1;
endfor

end