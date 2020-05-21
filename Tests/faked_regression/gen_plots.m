T = load('timings.txt');
P = (1:length(T))';

figure();
subplot(1,2,1);
plot(P, T(1)./T, 'k-*');
xlabel('Number of threads');
ylabel('Speedup relative to serial');
subplot(1,2,2);
plot(P, T(1) ./ (P.*T), 'k-*');
xlabel('Number of threads');
ylabel('Parallel Efficiency (as \% of linear)');

sgtitle('''Heavy Gradient'' Efficiency Study');
