close all;
clear all;

% load("../python_offline_analysis/output/debug/debug2.mat");


load("../output/tracking_results.mat")
n_idxs = size(est_positions, 1);
% n_idxs = 250;
tmp = squeeze(est_positions(1:n_idxs, : ,:));

figure();
plot3(tmp(:, 1), tmp(:, 2), tmp(:, 3));
xlim([-1, 1]);
ylim([-1, 1]);
zlim([-1, 1]);
pbaspect([1,1,1]);
xlabel("X (m)");
ylabel("Y (m)");
zlabel("Z (m)");

figure();
hold on;
plot(tmp(:, 1), tmp(:, 3), '.', 'markersize', 16);
plot(tmp(:, 1), tmp(:, 3));
xlim([-0.2, 2]);
ylim([-0.1, 1]);
pbaspect([2,1,1]);
title("x-z");
xlabel("X (m)");
ylabel("Z (m)");


figure('position', [200, 200, 1000, 300]);
subplot(1, 3, 1);
hold on;
plot(tmp(:, 2), tmp(:, 3), '.', 'markersize', 16);
plot(tmp(:, 2), tmp(:, 3));
xlim([-0.2, 2]);
ylim([-0.1, 1]);
pbaspect([2,1,1]);
title("y-z");
xlabel("Y (m)");
ylabel("Z (m)");

subplot(1, 3, 2);
hold on;
plot(tmp(:, 1), tmp(:, 3), '.', 'markersize', 16);
plot(tmp(:, 1), tmp(:, 3));
xlim([-0.2, 2]);
ylim([-0.1, 1]);
pbaspect([2,1,1]);
title("x-z");
xlabel("X (m)");
ylabel("Z (m)");

subplot(1, 3, 3);
hold on;
plot(tmp(:, 1), tmp(:, 2), '.', 'markersize', 16);
plot(tmp(:, 1), tmp(:, 2));
xlim([-0.1, 1]);
ylim([-0.1, 1]);
pbaspect([1,1,1]);
title("x-y");
xlabel("X (m)");
ylabel("Y (m)");
