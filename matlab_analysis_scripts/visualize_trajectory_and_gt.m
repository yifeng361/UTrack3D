close all;
clear all;

load("../output/tracking_results.mat");
true_positions_1 = gt_marker_positions(:, 7:9);
true_positions_1 = mocap23dtracku(true_positions_1);

% Visualize raw estimated trajectory and ground-truth trajectory.
% idxs1 = 1:500;
% idxs2 = 1:2800;
visualize_estimated_and_true_trajectory(est_positions, true_positions_1);

function new_positions = mocap23dtracku(positions)
    x = positions(:, 1);
    y = positions(:, 2);
    z = positions(:, 3);
    new_positions = [z,  -x, y];
end

function visualize_estimated_and_true_trajectory(est_positions, true_positions)
    n_idxs = size(est_positions, 1);
    tmp = squeeze(est_positions(1:n_idxs, : ,:));
    
    true_pos = true_positions;
    c = mean(est_positions);
    
    figure();
    hold on;
    plot3(tmp(:, 1), tmp(:, 2), tmp(:, 3));
    plot3(true_pos(:, 1), true_pos(:, 2), true_pos(:, 3));
    xlim([c(1)-1, c(1)+1]);
    ylim([c(2)-1, c(2)+1]);
    zlim([c(3)-1, c(3)+1]);
    pbaspect([1,1,1]);
    xlabel("X (m)");
    ylabel("Y (m)");
    zlabel("Z (m)");

    figure();
    hold on;
%     plot(tmp(:, 1), tmp(:, 3), '.', 'markersize', 16);
    plot(tmp(:, 1), tmp(:, 3));
    plot(tmp(1, 1), tmp(1, 3), '.', 'markersize', 16, 'color', 'k');
    plot(true_pos(:, 1), true_pos(:, 3));
    plot(true_pos(1, 1), true_pos(1, 3), '.', 'markersize', 16, 'color', 'k');
    xlim([c(1)-0.2, c(1) + 2]);
    ylim([c(2)-0.1, c(2) + 1]);
    pbaspect([2,1,1]);
    title("x-z");
    xlabel("X (m)");
    ylabel("Z (m)");
end