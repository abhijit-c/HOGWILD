clear alll; close all;

R = load('residuals.txt');

figure();
semilogy(400*(1:length(R)), R, 'k-*');
xlabel('Iteration Count')
ylabel('Loss function value');
title('Dense matrix regression convergence');
