A = csvread('simplemat.csv');
b = A(:,1);
A = A(:,2:end);

X = csvread('results.txt');

x_star = A\b;

fprintf('Least squares residual: %.4f\n', norm(A*x_star-b));
disp('Computed residuals for each processor count');
vecnorm(A*X-b)'
