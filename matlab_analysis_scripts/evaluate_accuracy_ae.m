close all;
clear all;

% Load estimated trajectory and ground-truth trajectory.
load("../output/tracking_results.mat");
true_positions_1 = gt_marker_positions(:, 7:9);
true_positions_1 = mocap23dtracku(true_positions_1);

% Process estimated and ground-truth trajectory
% (1) coarse axis aligning on ground-truth trajectory
% (2) Resample ground truth trajectory. Make it a multiple of the estimated
% trajectory's sampling rate.
% (3) remove static idxs at the beginning and the end.
est_skip_start = align_indices(1);
est_skip_end = align_indices(2);
true_skip_start = align_indices(3);
true_skip_end = align_indices(4);
upsample_rate = 2; % MoCap is 120Hz, UWB is 80Hz. 120 x 2 = (3) x 80.
skips = [est_skip_start, est_skip_end, true_skip_start, true_skip_end];
[new_est, new_true] = process_trajectory(est_positions, true_positions_1, ...
     skips);
diff = mean(new_est) - mean(new_true);
new_true_zero_bias = new_true + diff;

new_est(:, 1) = (new_est(:, 1) - mean(new_est(:, 1))) + mean(new_est(:, 1));

% These idxs don't account into errors. This is to avoid too many
% stationary idxs influence overall look.
visualize_estimated_and_true_trajectory(new_est, new_true);

eidxs = 1: size(new_est, 1);
tidxs = 1: size(new_true, 1);
[aligned_est, aligned_true] = align_trajectory_min_error(new_est, new_true,...
    eidxs, tidxs);

errors = compute_error(aligned_true, aligned_est);
figure();
cdfplot(errors);
xlabel("Error(m)");
saveas(gcf,'cdf.png')

% Save output data.
raw_est_positions = est_positions;
raw_true_positions = true_positions_1;

save("./results/tmp.mat", "raw_est_positions", ...
    "raw_true_positions", "skips", "est_skip_start", "est_skip_end", ...
    "true_skip_start", "true_skip_end", "upsample_rate", "aligned_est", "aligned_true", ...
    "errors", "start_position");

% Uncomment it only when you need very detailed data
% save("./data/evaluation_results/detailed_info.mat", "seq_cirs", "");

function [aligned_est, aligned_true] = align_trajectory_min_error(est_pos, true_pos,...
    eidxs, tidxs)

    center_Y = mean(true_pos(tidxs,:));
    center_X = mean(est_pos(eidxs, :));
    Y = true_pos - center_Y;
    X = est_pos - center_X;
    
    visualize_estimated_and_true_trajectory(X(eidxs,:), Y(tidxs,:));
    
    R = bruteforce_R(Y(tidxs, :), X(eidxs,:));
    
    
    
    Yhat = (R*X')';
    visualize_estimated_and_true_trajectory(Yhat, Y);
    aligned_est = Yhat;
    aligned_true = Y;
end


function [new_est, new_true] = process_trajectory(est_pos, true_pos, skips)
    % Skip some starting and ending idxs.
    est_skip_start = skips(1);
    est_skip_end = skips(2);
    true_skip_start = skips(3);
    true_skip_end = skips(4);
    
    n = size(est_pos, 1);
    new_est = est_pos(1 + est_skip_start: n - est_skip_end, :);
    n = size(true_pos, 1);
    new_true = true_pos(1 + true_skip_start: n - true_skip_end, :);

end

function new_true = resample_true(true_pos, upsample_rate)% Re-sample.
    step = 1 / upsample_rate;
    [n_samples, n_dim] = size(true_pos);
    for i = 1: n_dim
        new_true(:, i) = interp1(1:n_samples, true_pos(:, i), 1:step:n_samples);
    end
end

function visualize_estimated_trajectory_only(est_positions)
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
end

function visualize_estimated_and_true_trajectory(est_positions, true_positions)
    n_idxs = size(est_positions, 1);
    tmp = squeeze(est_positions(1:n_idxs, : ,:));
    
    true_pos = true_positions;
    c = mean(est_positions);
    
    figure();
    hold on;
    p1 = plot3(tmp(:, 1), tmp(:, 2), tmp(:, 3));
    p2 = plot3(true_pos(:, 1), true_pos(:, 2), true_pos(:, 3));
    xlim([c(1)-0.5, c(1)+0.5]);
    ylim([c(2)-0.5, c(2)+0.5]);
    zlim([c(3)-0.5, c(3)+0.5]);
    pbaspect([1,1,1]);
    xlabel("X (m)");
    ylabel("Y (m)");
    zlabel("Z (m)");
    legend([p1, p2], ["Estimated", "True"]);
    view(3);
    saveas(gcf,'viz3d.png')

    figure();
    hold on;
%     plot(tmp(:, 1), tmp(:, 3), '.', 'markersize', 16);
    p1 = plot(tmp(:, 1), tmp(:, 3));
    plot(tmp(1, 1), tmp(1, 3), '.', 'markersize', 16, 'color', 'k');
    p2 =  plot(true_pos(:, 1), true_pos(:, 3));
    plot(true_pos(1, 1), true_pos(1, 3), '.', 'markersize', 16, 'color', 'k');
    xlim([c(1)-0.5, c(1) + 0.5]);
    ylim([c(2)-0.5, c(2) + 0.5]);
    pbaspect([1,1,1]);
    title("x-z");
    xlabel("X (m)");
    ylabel("Z (m)");
    legend([p1, p2], ["Estimated", "True"]);
    saveas(gcf,'viz2d.png')
end

% Has to be already aligned: Y = RX
function optR = bruteforce_R(Y, X)
    xs = -5:1:5;
    ys = -5:1:5;
    zs = -5:1:5;

    minError = 1e9;
    optijk = [1,1,1];
    for i = 1: length(xs)
        for j = 1:length(ys)
            for k = 1: length(zs)
                tic;
                x = xs(i); y = ys(j); z = zs(k);
                R = eul2rotm(deg2rad([x,y,z]));
                Yhat = (R * X')';
                error_vec = zeros(size(Yhat, 1), 1);
                parfor u = 1: size(Yhat, 1)
                    error_vec(u) = min(vecnorm(Y-Yhat(u,:), 2, 2));
                end
                error = mean(error_vec);
                if (error < minError)
                    minError = error;
                    optijk = [i,j,k];
                end
                toc;
            end
        end
    end
    optR = eul2rotm(deg2rad([xs(optijk(1)), ys(optijk(2)), zs(optijk(3))]));

end

function new_positions = mocap23dtracku(positions)
    x = positions(:, 1);
    y = positions(:, 2);
    z = positions(:, 3);
    new_positions = [z,  -x, y];
end

function errors = compute_error(true_pos, est_pos)
    errors = zeros(size(est_pos, 1), 1);
    for i = 1: size(est_pos, 1)
        errors(i) = min(vecnorm(true_pos-est_pos(i,:), 2, 2));
    end
end