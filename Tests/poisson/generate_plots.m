clear all; close all;
N = 2^20;
x = linspace(0, 2*pi, N)';

e = ones(N, 1);
A = (1/(x(2)-x(1))^2)*spdiags([e -2*e e], -1:1, N, N);
f = -sin(x);

tic; sol = A \ f; toc;

S = csvread('results.txt');

figure();
hold on;
plot(x, sin(x), 'k-');
plot(linspace(0,2*pi,length(S)), S, 'k--');
hold off;
