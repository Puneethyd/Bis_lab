clear; clc; close all;
rng(1);

start  = [1 1];
target = [9 9];
obs = [3.0 3.0 1.0;
       5.0 6.0 1.2;
       7.5 4.0 1.0;
       4.0 8.5 0.8;
       8.0 8.0 0.6;
       2.0 6.0 0.7];

P.stepSize    = 0.40;
P.sensorRange = 1.50;
P.safeDist    = 0.50;
P.goalTol     = 0.30;
P.maxSteps    = 300;
P.nParticles  = 25;
P.nIter       = 15;
P.w           = 0.7;
P.c1          = 1.5;
P.c2          = 1.5;
P.kObs        = 1.0;
P.kTurn       = 0.30;
P.kVisit      = 0.50;

[path1, col1] = runRobot(1, start, target, obs, P);
[path2, col2] = runRobot(2, start, target, obs, P);

len1 = pathLength(path1);
len2 = pathLength(path2);
reached1 = norm(path1(end,:) - target) < P.goalTol;
reached2 = norm(path2(end,:) - target) < P.goalTol;

fprintf('\n=========== RESULTS ===========\n');
fprintf('%-10s %-10s %-12s %-8s %-10s\n', 'Planner', 'Reached', 'Length(m)', 'Steps', 'Collisions');
fprintf('%-10d %-10d %-12.3f %-8d %-10d\n', 1, reached1, len1, size(path1,1)-1, col1);
fprintf('%-10d %-10d %-12.3f %-8d %-10d\n', 2, reached2, len2, size(path2,1)-1, col2);

figure('Name', 'PSO Target Seeking Robot', 'Position', [100 100 1100 480]);
titles = {'Planner 1: distance + obstacle', 'Planner 2: + smoothness + memory'};
paths  = {path1, path2};
for i = 1:2
    subplot(1, 2, i); hold on; axis equal; grid on;
    xlim([0 10]); ylim([0 10]);
    for k = 1:size(obs,1)
        rectangle('Position', [obs(k,1)-obs(k,3), obs(k,2)-obs(k,3), 2*obs(k,3), 2*obs(k,3)], ...
                  'Curvature', [1 1], 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
    end
    plot(paths{i}(:,1), paths{i}(:,2), 'b-o', 'MarkerSize', 3, 'LineWidth', 1.3);
    plot(start(1),  start(2),  'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(target(1), target(2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
    title(titles{i}); xlabel('x (m)'); ylabel('y (m)');
    legend({'Path', 'Start', 'Target'}, 'Location', 'southeast');
end

function [path, collisions] = runRobot(planner, start, target, obs, P)
    pos        = start;
    heading    = atan2(target(2)-start(2), target(1)-start(1));
    path       = pos;
    collisions = 0;

    for step = 1:P.maxSteps
        if norm(pos - target) < P.goalTol
            break;
        end

        fit = @(d) fitness(pos + d, pos, heading, target, obs, path, P, planner);

        X = randDisp(P.nParticles, P.stepSize);
        V = zeros(P.nParticles, 2);
        pbest  = X;
        pbestF = zeros(P.nParticles, 1);
        for i = 1:P.nParticles
            pbestF(i) = fit(X(i,:));
        end
        [gbestF, idx] = min(pbestF);
        gbest = pbest(idx, :);

        for it = 1:P.nIter
            for i = 1:P.nParticles
                r1 = rand(1,2); r2 = rand(1,2);
                V(i,:) = P.w*V(i,:) ...
                       + P.c1*r1.*(pbest(i,:) - X(i,:)) ...
                       + P.c2*r2.*(gbest      - X(i,:));
                X(i,:) = X(i,:) + V(i,:);

                n = norm(X(i,:));
                if n > P.stepSize
                    X(i,:) = X(i,:) * P.stepSize / n;
                end

                f = fit(X(i,:));
                if f < pbestF(i)
                    pbestF(i) = f;
                    pbest(i,:) = X(i,:);
                    if f < gbestF
                        gbestF = f;
                        gbest  = X(i,:);
                    end
                end
            end
        end

        d = gbest;
        if norm(d) > 1e-6
            heading = atan2(d(2), d(1));
        end
        pos  = pos + d;
        path = [path; pos];

        for k = 1:size(obs,1)
            if norm(pos - obs(k,1:2)) < obs(k,3)
                collisions = collisions + 1;
                break;
            end
        end
    end
end

function D = randDisp(n, stepSize)
    ang = 2*pi*rand(n,1);
    rad = stepSize*sqrt(rand(n,1));
    D   = [rad.*cos(ang), rad.*sin(ang)];
end

function f = fitness(p, pos, heading, target, obs, path, P, planner)
    f = norm(p - target);

    pen = 0;
    for k = 1:size(obs,1)
        c = obs(k,1:2);  r = obs(k,3);
        if norm(pos - c) - r <= P.sensorRange
            ds = norm(p - c) - r;
            if ds < 0
                pen = pen + 1000;
            elseif ds < P.safeDist
                pen = pen + P.kObs*(1/max(ds,0.05) - 1/P.safeDist);
            end
        end
    end
    f = f + pen;

    if planner == 2
        d = p - pos;
        if norm(d) > 1e-6
            newH = atan2(d(2), d(1));
            dH   = abs(atan2(sin(newH-heading), cos(newH-heading)));
            f = f + P.kTurn*dH;
        end
        if size(path,1) > 5
            recent = path(1:end-3, :);
            nVisit = sum(sqrt(sum((recent - p).^2, 2)) < 0.25);
            f = f + P.kVisit*nVisit;
        end
    end
end

function L = pathLength(path)
    L = sum(sqrt(sum(diff(path).^2, 2)));
end
