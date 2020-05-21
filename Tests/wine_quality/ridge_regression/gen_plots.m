Data = load('winequality-white.csv');

[N,M] = size(Data);
A = Data(:,1:end-1);
b = Data(:,end);

f = @(x_star) (0.5/N)*norm(A*x_star-b)^2;

cvx_begin
  variable x(M-1)
  minimize ( (0.5/N)*square_pos(norm(A*x-b)) + 3*norm(x) )
cvx_end

X = load('X.txt');

fprintf('True residual: %.5f\n Computed residual: %.5f\n', f(x), f(X));

R = load('results.txt');

figure();
hold on;
plot(1:length(R), R/N, 'k-*');
plot(1:length(R), f(x)*ones(size(R)), 'k--');
hold off;
set(gca, 'yscale', 'log');
xlabel('Epoch count');
ylabel('Losses');
legend('Losses during computation', 'True solution');
title('Wine quality ridge regression');
