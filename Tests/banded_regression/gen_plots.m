V = load('variances.txt');

figure();
loglog(2.^(0:length(V)-1), V, 'k-*');
title('Asynchronous Noise as a function of bandedness');
xlabel('Length of Band (as powers of two)');
ylabel('Computed Variance');
