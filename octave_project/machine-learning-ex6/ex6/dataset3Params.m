function [C, sigma] = dataset3Params(X, y, Xval, yval)

% 피팅밸류리스트
fittingValueList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n = size(fittingValueList, 2);

minC = fittingValueList(1);
minSigma = fittingValueList(1);
model= svmTrain(X, y, minC, @(x1, x2) gaussianKernel(x1, x2, minSigma));
predictions = svmPredict(model, Xval);
minMean = mean(double(predictions ~= yval));

for i = 1:n
  for j = 1:n
    C = fittingValueList(i);
    sigma = fittingValueList(j);
    % 트레이닝셋으로 트레이닝
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    % 얻은 결과를 밸류셋으로 평가
    predictions = svmPredict(model, Xval);
    % 평가가 제일 좋은 경우의 피팅밸류 찾기
    if minMean > mean(double(predictions ~= yval))
      minMean = mean(double(predictions ~= yval));
      minC = C;
      minSigma = sigma;
    end
  end
end  

C = minC;
sigma = minSigma;      
end