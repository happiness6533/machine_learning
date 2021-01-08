function plotData(X, y)
  % 그래프 오픈
  figure;
  
  hold on;

  % 원하는 행 찾기
  pos = find(y == 1);
  neg = find(y == 0);
  
  % 점 찍기
  plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
  plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  
  hold off;
end