clear alll; close all;

t = load('timings.txt')
P = (1:length(t))';

figure();

subplot(1,2,1);
plot(P, t, 'k-*');
xlabel('Number of threads');
ylabel('Time taken');

subplot(1,2,2);
plot(P, t(1) ./ (P.*t), 'k-*');
xlabel('Number of threads');
ylabel('Parallel Efficiency (as \% of linear)');

sgtitle('Large regression scaling study w/o replacement');
