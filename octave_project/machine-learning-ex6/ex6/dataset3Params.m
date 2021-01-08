function [C, sigma] = dataset3Params(X, y, Xval, yval)

% ���ù������Ʈ
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
    % Ʈ���̴׼����� Ʈ���̴�
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    % ���� ����� ��������� ��
    predictions = svmPredict(model, Xval);
    % �򰡰� ���� ���� ����� ���ù�� ã��
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