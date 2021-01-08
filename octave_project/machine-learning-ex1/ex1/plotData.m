% function 함수(리턴할 값이 없는 경우)
function plotData(x, y)
  figure;
  plot(x, y, 'rx', 'MarkerSize', 10);
  ylabel('Profit in $10,000s');
  xlabel('Population of City in 10,000s');
end