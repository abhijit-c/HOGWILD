Data = load('simplemat.csv');
A = Data(:,2:end);
b = Data(:,1);

x = A\b;
f = @(A,b,x) 0.5*norm(A*x-b)^2;

disp(f(A,b,x));
disp(f(A,b,x)/length(A));
