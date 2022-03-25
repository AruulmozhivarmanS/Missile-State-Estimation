clc;
close all;
clear;

tau = 2;
Y_0_MEAN = 0;
V_0_MEAN = 0;
At_0_MEAN = 0;

V_VAR = 200^2;
At_VAR = 100^2;

Vc = 300;
Tf = 10;
R1 = 15e-6;
R2 = 1.67e-3;

AtTlg = 100;

p_0 = zeros(3, 3);
p_0(2, 2) = V_VAR;
p_0(3, 3) = At_VAR;

dt = 0.01;

ts = linspace(0, Tf, Tf / dt + 1);
ts = ts(1:end-1);

F = [0, 1, 0; 0, 0, -1; 0, 0, -1/tau];
G = [0; 0; 1];
W = At_VAR;

[filterKs, filterPs] = computePGM(ts, p_0, F, G, W);

n = 1000;
trueXHistory= zeros(n, 3, size(ts, 2));
trueYHistory= zeros(n, 1, size(ts, 2));
xHatHistory= zeros(n, 3, size(ts, 2));
residualHistory = zeros(n, 1, size(ts, 2));

trueXHistoryRTS = zeros(n, 2, size(ts, 2));
trueYHistoryRTS = zeros(n, 1, size(ts, 2));
xHatHistoryRTS = zeros(n, 3, size(ts, 2));
residualHistoryRTS = zeros(n, 1, size(ts, 2));
trueAtHistoryRTS = zeros(n, 1, size(ts, 2));

for i = 1:n
    [trueXs, trueYs] = simDynamics(ts, F, G, W);
    trueXHistory(i, :, :) = trueXs;
    trueYHistory(i, :, :) = trueYs;
    [xHats, residuals] = myKfInnovate(ts, filterKs, trueYs, F);
    xHatHistory(i, :, :) = xHats;
    residualHistory(i, :, :) = residuals;

    [trueXsRTS, trueYsRTS, atRTS] = simTelegraphDynamics(dt, ts, AtTlg, Tf);
    trueXHistoryRTS(i, :, :) = trueXsRTS;
    trueYHistoryRTS(i, :, :) = trueYsRTS;
    trueAtHistoryRTS(i, :, :) = atRTS;
    [xHatsRTS, residualsRTS] = myKfInnovate(ts, filterKs, trueYsRTS, F);
    xHatHistoryRTS(i, :, :) = xHatsRTS;
    residualHistoryRTS(i, :, :) = residualsRTS;
end

[trueErrorVariance, meanError] = monteCarloAnalysis(trueXHistory, xHatHistory);
[trueErrorVarianceRTS, meanErrorRTS] = monteCarloAnalysis(cat(2, trueXHistoryRTS, trueAtHistoryRTS) , xHatHistoryRTS);

plotResults1(ts, Tf, filterKs, filterPs, trueErrorVariance)
plotResults2(ts, trueYs, trueXs, xHats, filterPs)
plotResults3(ts, trueYsRTS, cat(1, trueXsRTS, atRTS), xHatsRTS, filterPs)
plotResults4(ts, trueErrorVariance, trueErrorVarianceRTS, filterPs)
plotResults5(ts, dt, residualHistory)

figure(22)
plot(ts, meanError)
xlabel('Time Since Launch (s)')
ylabel('Error in Position (ft)')
set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
set(get(gca, 'Title'), 'String', 'Mean Error in State Estimates (Random Telegraph Signal)');

figure(23)
plot(ts, meanErrorRTS)
xlabel('Time Since Launch (s)')
ylabel('Error in Position (ft)')
set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
set(get(gca, 'Title'), 'String', 'Mean Error in Position Estimates (Random Telegraph Signal)');

function [ks, ps] = computePGM(ts, p_0, F, G, W)
    dt = ts(2) - ts(1);
    ps = zeros(1, 3, 3);
    ps(1, :, :) = p_0;

    ks = zeros(3, 1);
    ks(:, 1) = [0; 0; 0];
    
    R1 = 15e-6;
    R2 = 1.67e-3;
    Vc = 300;
    Tf = 10;

    for t = ts
        V = R1 + R2/((Tf-t)^2);

        H = [1/(Vc*(Tf - t)), 0, 0];
        ks(:, end+1) = squeeze(ps(end, : , :)) * H' / V;
        
        p_dot = F * squeeze(ps(end, :, :)) + squeeze(ps(end, : , :)) * F' - ks(:, end) * H * squeeze(ps(end, :, :)) + G * W * G';
        ps(end+1, :, :) = squeeze(ps(end, : ,:)) + p_dot * dt;
    end

    ks = ks(:, 2:end);
    ps = ps(2:end, :, :);

end

function [xs, ys] = simDynamics(ts, F, G, W) 
    dt = ts(2) - ts(1);
    
    R1 = 15e-6;
    R2 = 1.67e-3;
    Vc = 300;
    Tf = 10;

    Y_0_MEAN = 0;

    V_0_MEAN = 0;
    V_VAR = 200^2;

    At_0_MEAN = 0;
    At_VAR = 100^2;

    xs = [[Y_0_MEAN; normrnd(V_0_MEAN, sqrt(V_VAR)); normrnd(At_0_MEAN, sqrt(At_VAR))]];
    ys = [0];

    w_at = normrnd(0, sqrt(W/dt), size(ts));
    i = 1;
    V = R1 + R2./((Tf-ts).^2);
    v = normrnd(0, sqrt(V/dt));
    for t = ts
        H = [1/(Vc*(Tf - t)), 0, 0];
        x_dot = F * xs(:, end) + G * w_at(i);
        xs(:, end+1) = xs(:, end) + x_dot*dt;
        ys(end+1) = H * xs(:, end) + v(i);
        i = i + 1;
    end

    xs = xs(:, 2:end);
    ys = ys(2:end);
end

function [xHats, residuals] = myKfInnovate(ts, ks, zs, F)
    dt = ts(2) - ts(1);
    Vc = 300;
    Tf = 10;
    
    xHats = [[0; 0; 0]];
    residuals = [0];
    i = 1;
    for t = ts
        H = [1/(Vc*(Tf - t)), 0, 0];
        residual = zs(i) - H * xHats(:, end);
        x_dot = F*xHats(:, end) + ks(:, i) * residual;
        xHats(:, end+1) = xHats(:, end) + x_dot * dt;
        residuals(end+1) = residual;
        i = i + 1;
    end

    xHats = xHats(:, 2:end);
    residuals = residuals(2:end);
end

function [trueErrorVariance, meanError] = monteCarloAnalysis(trueXHistory, xHatHistory)
    error = trueXHistory - xHatHistory;
    meanError = squeeze(mean(error, 1));
    trueErrorVariance = zeros(size(xHatHistory, 3), 3, 3);
    for i = 1:size(xHatHistory, 3)
        for j = 1:size(xHatHistory, 1)
            trueErrorVariance(i, :, :) = squeeze(trueErrorVariance(i, :, :)) + ((error(j, :, i)' - meanError(:, i)) * ((error(j, :, i)' - meanError(:, i))'));
        end
    end
    trueErrorVariance = trueErrorVariance / (size(xHatHistory, 1)-1);

end

function aRTS = generateRTS(dt, Tf, AtTlg)
    t = 0;
    ts = [];
    lambda = 0.25;
    while t < Tf
        t = t - log(unifrnd(0, 1))/lambda;
        ts(end+1) = t;
    end

    ts = unique(roundn(ts, log10(dt)));

    sampleT = linspace(0, Tf, Tf / dt + 1);

    [locA, locB] = ismember(ts, sampleT);
    locB = locB(locB ~= 0);

    switchT = zeros(size(sampleT));
    switchT(locB) = 1;

    aRTS = zeros(size(sampleT)-1);
    flip = binornd(1, 0.5);
    if(flip)
        sign = 1;
    else
        sign = -1;
    end

    for i = 1:size(aRTS, 2)
        if(switchT(i) == 1)
            sign = sign * -1;
        end
        aRTS(i) = sign * AtTlg;
    end
end

function [xs, ys, atRTS] = simTelegraphDynamics(dt, ts, AtTlg, Tf)
    atRTS = generateRTS(dt, Tf, AtTlg);
    F = [0, 1; 0, 0];
    G = [0; -1];

    dt = ts(2) - ts(1);
    
    R1 = 15e-6;
    R2 = 1.67e-3;
    Vc = 300;
    Tf = 10;

    Y_0_MEAN = 0;

    V_0_MEAN = 0;
    V_VAR = 200^2;

    xs = [[Y_0_MEAN; normrnd(V_0_MEAN, sqrt(V_VAR))]];
    ys = [0];
    
    i = 1;
    V = R1 + R2./((Tf-ts).^2);
    v = normrnd(0, sqrt(V/dt));
    for t = ts
        H = [1/(Vc*(Tf - t)), 0];
        x_dot = F * xs(:, end) + G * atRTS(i);
        xs(:, end+1) = xs(:, end) + x_dot*dt;
        ys(end+1) = H * xs(:, end) + v(i);
        i = i + 1;
    end

    xs = xs(:, 2:end);
    ys = ys(2:end);
end

function residualCorrelations = residualAnalysis(timeToAnalysis, residualHistory, dt)
    residualT = residualHistory(:, :, (timeToAnalysis/dt)+1);
    residualCorrelations = zeros(size(residualHistory, 2), size(residualHistory, 3));
    for i = 1:size(residualHistory, 1)
        for j = 1:size(residualHistory, 3)
            residualCorrelations(j) = residualCorrelations(j) + (residualHistory(i, 1, j) * residualT(i, 1)');
        end
    end
    residualCorrelations = residualCorrelations / size(residualHistory, 1);
end

function plotResults1(ts, Tf, filterKs, filterPs, trueErrorVariance)
    figure(1)
    ax1 = axes;
    plot(Tf - ts, filterKs(1, :))
    hold on
    plot(Tf - ts, filterKs(2, :))
    plot(Tf - ts, filterKs(3, :))
    hold off
    set(ax1, 'Xdir', 'reverse')
    xlabel('Time-to-Go (s)')
    ylabel('Kalman Filter Gains')
    legend('K_{1}', 'K_{2}', 'K_{3}')
    set(get(gca, 'Title'), 'String', 'Kalman Filter Gains');

    figure(2)
    ax1 = axes;
    plot(Tf - ts, sqrt(filterPs(:, 1, 1)))
    hold on
    plot(Tf - ts, sqrt(filterPs(:, 2, 2)))
    plot(Tf - ts, sqrt(filterPs(:, 3, 3)))
    plot(Tf - ts, sqrt(trueErrorVariance(:, 1, 1)))
    plot(Tf - ts, sqrt(trueErrorVariance(:, 2, 2)))
    plot(Tf - ts, sqrt(trueErrorVariance(:, 3, 3)))
    hold off
    set(ax1, 'Xdir', 'reverse')
    xlabel('Time-to-Go (s)')
    ylabel('Standard Deviation of State Error')
    legend('\sigma_{y}', '\sigma_{v}', '\sigma_{a_{T}}', 'Actual \sigma_{y}', 'Actual \sigma_{v}', 'Actual \sigma_{a_{T}}')
    set(get(gca, 'Title'), 'String', 'Kalman Filter Estimated State RMS Error P(t)^{1/2} vs. Actual State RMS Error');

    figure(21)
    ax1 = axes;
    plot(Tf - ts, filterPs(:, 1, 1))
    hold on
    plot(Tf - ts, filterPs(:, 2, 2))
    plot(Tf - ts, filterPs(:, 3, 3))
    hold off
    set(ax1, 'Xdir', 'reverse')
    xlabel('Time-to-Go (s)')
    ylabel('Estimates of State Error Variance')
    legend('\sigma_{y}^{2}', '\sigma_{v}^{2}', '\sigma_{a_{T}}^{2}')
    set(get(gca, 'Title'), 'String', 'Kalman Filter State Error Variance P(t)');

end

function plotResults2(ts, trueYs, trueXs, xHats, filterPs)
    figure(3)
    plot(ts, trueYs)
    xlabel('Time Since Launch (s)')
    ylabel('\theta (rad)')
    set(gca, 'YScale', 'log')
    set(get(gca, 'Title'), 'String', 'Measurement \theta (Gauss-Markov Process)');

    figure(4)
    plot(ts, trueXs(1, :))
    hold on
    plot(ts, xHats(1, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Position (ft)')
    set(legend('$y$', '$\hat{y}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Position vs. Estimated Position (Gauss-Markov Process)');

    figure(5)
    plot(ts, trueXs(2, :))
    hold on
    plot(ts, xHats(2, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Velocity (ft/s)')
    set(legend('$v$', '$\hat{v}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Velocity vs. Estimated Velocity (Gauss-Markov Process)');

    figure(6)
    plot(ts, trueXs(3, :))
    hold on
    plot(ts, xHats(3, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Acceleration (ft/s^2)')
    set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Acceleration vs. Estimated Acceleration (Gauss-Markov Process)');

    figure(7)
    P1 = sqrt(filterPs(:, 1, 1)');
    plot(ts, trueXs(1, :) - xHats(1, :))
    hold on
    plot(ts, P1, 'black')
    plot(ts, - P1, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Position (ft)')
    set(get(gca, 'Title'), 'String', 'Error in Position Estimates with 1-\sigma (Gauss-Markov Process)');

    figure(8)
    P2 = sqrt(filterPs(:, 2, 2)');
    plot(ts, trueXs(2, :) - xHats(2, :))
    hold on
    plot(ts, P2, 'black')
    plot(ts, - P2, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Velocity (ft/s)')
    set(get(gca, 'Title'), 'String', 'Error in Velocity Estimates with 1-\sigma (Gauss-Markov Process)');

    figure(9)
    P3 = sqrt(filterPs(:, 3, 3)');
    plot(ts, trueXs(3, :) - xHats(3, :))
    hold on
    plot(ts, P3, 'black')
    plot(ts, - P3, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Acceleration (ft/s^2)')
    set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Error in Acceleration Estimates with 1-\sigma (Gauss-Markov Process)');

end

function plotResults3(ts, trueYs, trueXs, xHats, filterPs)
    figure(10)
    plot(ts, trueYs)
    xlabel('Time Since Launch (s)')
    ylabel('\theta (rad)')
    % set(gca, 'YScale', 'log')
    set(get(gca, 'Title'), 'String', 'Measurement \theta (Random Telegraph Signal)');

    figure(11)
    plot(ts, trueXs(1, :))
    hold on
    plot(ts, xHats(1, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Position (ft)')
    set(legend('$y$', '$\hat{y}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Position vs. Estimated Position (Random Telegraph Signal)');

    figure(12)
    plot(ts, trueXs(2, :))
    hold on
    plot(ts, xHats(2, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Velocity (ft/s)')
    set(legend('$v$', '$\hat{v}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Velocity vs. Estimated Velocity (Random Telegraph Signal)');

    figure(13)
    plot(ts, trueXs(3, :))
    hold on
    plot(ts, xHats(3, :))
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Acceleration (ft/s^2)')
    set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Actual Acceleration vs. Estimated Acceleration (Random Telegraph Signal)');

    figure(14)
    P1 = sqrt(filterPs(:, 1, 1)');
    plot(ts, trueXs(1, :) - xHats(1, :))
    hold on
    plot(ts, P1, 'black')
    plot(ts, - P1, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Position (ft)')
    set(get(gca, 'Title'), 'String', 'Error in Position Estimates with 1-\sigma (Random Telegraph Signal)');

    figure(15)
    P2 = sqrt(filterPs(:, 2, 2)');
    plot(ts, trueXs(2, :) - xHats(2, :))
    hold on
    plot(ts, P2, 'black')
    plot(ts, - P2, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Velocity (ft/s)')
    set(get(gca, 'Title'), 'String', 'Error in Velocity Estimates with 1-\sigma (Random Telegraph Signal)');

    figure(16)
    P3 = sqrt(filterPs(:, 3, 3)');
    plot(ts, trueXs(3, :) - xHats(3, :))
    hold on
    plot(ts, P3, 'black')
    plot(ts, - P3, 'black')
    hold off
    xlabel('Time Since Launch (s)')
    ylabel('Error in Acceleration (ft/s^2)')
    set(legend('$a_{T}$', '$\hat{a_{T}}$'),'Interpreter','Latex')
    set(get(gca, 'Title'), 'String', 'Error in Acceleration Estimates with 1-\sigma (Random Telegraph Signal)');

end

function plotResults4(ts, trueErrorVariance, trueErrorVarianceRTS, filterPs)
    figure(17)
    subplot(3, 3, 1)
    hold on
    plot(ts, trueErrorVariance(:, 1, 1))
    plot(ts, filterPs(:, 1, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 1)')

    subplot(3, 3, 2)
    hold on
    plot(ts, trueErrorVariance(:, 1, 2))
    plot(ts, filterPs(:, 1, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 2)')

    subplot(3, 3, 3)
    hold on
    plot(ts, trueErrorVariance(:, 1, 3))
    plot(ts, filterPs(:, 1, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 3)')

    subplot(3, 3, 4)
    hold on
    plot(ts, trueErrorVariance(:, 2, 1))
    plot(ts, filterPs(:, 2, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 1)')

    subplot(3, 3, 5)
    hold on
    plot(ts, trueErrorVariance(:, 2, 2))
    plot(ts, filterPs(:, 2, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 2)')

    subplot(3, 3, 6)
    hold on
    plot(ts, trueErrorVariance(:, 2, 3))
    plot(ts, filterPs(:, 2, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 3)')

    subplot(3, 3, 7)
    hold on
    plot(ts, trueErrorVariance(:, 3, 1))
    plot(ts, filterPs(:, 3, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 1)')

    subplot(3, 3, 8)
    hold on
    plot(ts, trueErrorVariance(:, 3, 2))
    plot(ts, filterPs(:, 3, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 2)')

    subplot(3, 3, 9)
    hold on
    plot(ts, trueErrorVariance(:, 3, 3))
    plot(ts, filterPs(:, 3, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 3)')

    sgtitle('True Error Covariance vs. Estimated Error Covariance for 1000 Monte Carlo Runs (Gauss-Markov Process)')

    figure(18)
    subplot(3, 3, 1)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 1, 1))
    plot(ts, filterPs(:, 1, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 1)')

    subplot(3, 3, 2)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 1, 2))
    plot(ts, filterPs(:, 1, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 2)')

    subplot(3, 3, 3)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 1, 3))
    plot(ts, filterPs(:, 1, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(1, 3)')

    subplot(3, 3, 4)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 2, 1))
    plot(ts, filterPs(:, 2, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 1)')

    subplot(3, 3, 5)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 2, 2))
    plot(ts, filterPs(:, 2, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 2)')

    subplot(3, 3, 6)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 2, 3))
    plot(ts, filterPs(:, 2, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(2, 3)')

    subplot(3, 3, 7)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 3, 1))
    plot(ts, filterPs(:, 3, 1))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 1)')

    subplot(3, 3, 8)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 3, 2))
    plot(ts, filterPs(:, 3, 2))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 2)')

    subplot(3, 3, 9)
    hold on
    plot(ts, trueErrorVarianceRTS(:, 3, 3))
    plot(ts, filterPs(:, 3, 3))
    hold off
    xlabel('Time Since Launch (s)')
    set(legend('True P', 'P'),'Interpreter','Latex')
    title('Covariance(3, 3)')

    sgtitle('True Error Covariance vs. Estimated Error Covariance for 1000 Monte Carlo Runs (Random Telegraph Signal)')
end

function plotResults5(ts, dt, residualHistory)
    residualCorrelations = residualAnalysis(7, residualHistory, dt);

    figure (19)
    plot(ts, residualCorrelations)
    xlabel('Time Since Launch (s)')
    ylabel('Correlation')
    title('Correlation of Residual at 7^{th} second with Time')
    Tf = 10;
    R1 = 15e-6;
    R2 = 1.67e-3;
    V = R1 + R2./((Tf-ts).^2);

    figure (20)
    periodogram(squeeze(mean(residualHistory, 1))./V')
    title('Power Spectral Density of Residual Process')

end
