function X_rec = recoverData(Z, U, K)

% �������� �������� ����
% U���� = ���̰պ���, ���� ����� = transpose!
X_rec = Z * U(:, 1:K)';

end