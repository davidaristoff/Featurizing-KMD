%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% FKMD analysis script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load FKMD_iter1.mat

%% prepare plots

%choose collection of iterations to analyze
iterset = [1 2 3 4 5];

%define time step of input data
dt = 0.01;

%set plotting preferences
colors = set_plotting_preferences(iterset);

%% perform analysis for plots

%define length of iteration set
k = length(iterset);

%define observation matrix size
[steps,m] = size(obs_ref); 

%initialize data vectors
RMSE = zeros(steps,k);
corrs = zeros(steps,k);
Ms = zeros(d,d,k);
predictions = zeros(steps,m,k);
reference = zeros(steps,m,k);

%loop over iterations
for i=1:k

%get iteration number and load data
iter = iterset(i);
    load(['FKMD_iter',num2str(iter)], ...
        "obs_inf","obs_ref","M");

    %save predictions and reference
    predictions(:,:,i) = obs_inf; reference(:,:,i) = obs_ref;

    %trim noise from observations (FOR LORENZ EXPERIMENT ONLY!)
    obs_inf = obs_inf(:,1:2:end); obs_ref = obs_ref(:,1:2:end);

    %compute and save RMSE
    RMSE(:,i) = vecnorm(obs_inf-obs_ref,2,2) ...
                /(sqrt(m)*mean(vecnorm(obs_ref,2,2)));

    %compute and save correlations
    corrs(:,i) = diag(corr(obs_inf(:,1:2:end)',obs_ref(:,1:2:end)'));

    %compute and save scaling matrix
    Ms(:,:,i) = M;

end

%% load ordinary KMD simulation results

load KMD.mat obs_inf obs_ref

%% begin plotting

%plot initial and final predictions
figure('Position', 0.7*[30 30 800 400]);
subplot(1,2,1);
ts = 0:dt:(steps-1)*dt; 
plot(ts,reference(1:length(ts),end-1,1),'--b','linewidth',2); hold on; 
p = plot(ts,predictions(1:length(ts),end-1,1),'-k','linewidth',6);
p.Color(4) = 0.15;
xlabel('time'); ylabel('prediction');
xlim([min(ts) max(ts)]); ylim([-9 14]);
legend('reference','FKMD','location','northwest'); 
title("iteration 1");
subplot(1,2,2);
ts = 0:dt:(steps-1)*dt; 
plot(ts,reference(1:length(ts),end-1,end),'--b','linewidth',2); hold on; 
p = plot(ts,predictions(1:length(ts),end-1,end),'-k','linewidth',6);
p.Color(4) = 0.15;
xlabel('time'); ylabel('prediction');
xlim([min(ts) max(ts)]); ylim([-9 14]);
legend('reference','FKMD','location','northwest'); 
title("iteration " + k);
sgtitle('predictions by iteration','fontsize',20);
saveas(gcf,'inference','pdf');

%plot RMSE
figure('Position', [30 30 460 400]); 
for i=1:k
    test_frames = 1:1:steps;
    xs = 0:.05:steps; 
    plot(test_frames,RMSE(:,i),'-','linewidth',2,'color',colors(i,:));
    hold on;
end
xlabel('test frame'); ylabel('relative RMS error');
legend('iteration 1', 'iteration 2', 'iteration 3',...
    'iteration 4','iteration 5','location','northwest');
title('prediction error to test set by iteration');
saveas(gcf,'RMSE','epsc');

%plot correlations
figure('Position', [30 30 460 400]); 
for i=1:k
    test_frames = 1:1:steps;
    xs = 0:.05:steps; 
    plot(test_frames,corrs(:,i),'-','linewidth',2,'color',colors(i,:));
    hold on;
end
xlabel('test frame'); ylabel('correlation coefficient');
legend('iteration 1', 'iteration 2', 'iteration 3',...
    'iteration 4','iteration 5','location','southwest');
title('prediction correlation to test set by iteration');
saveas(gcf,'corrs','epsc');

%plot scaling matrix by iteration
figure('Position', [30 30 800 220]);
subplot(1,3,1);
imagesc(Ms(end-19:end,end-19:end,1)); colorbar; 
title("iteration 1");
subplot(1,3,2);
imagesc(Ms(end-19:end,end-19:end,2)); colorbar; 
title("iteration 2");
subplot(1,3,3);
imagesc(Ms(end-19:end,end-19:end,3)); colorbar; 
title("iteration 3");
sgtitle('\mbox{{\boldmath $M$}$^{1/2}$} by iteration','fontsize',24);
saveas(gcf,'scaling_matrix','epsc');

%plot KMD result
load KMD.mat
figure('Position', [30 30 265 240]);
ts = 0:dt:(steps-1)*dt; 
plot(ts,obs_ref(1:length(ts),1),'--b','linewidth',2); hold on; 
p = plot(ts,obs_inf(1:length(ts),1),'-k','linewidth',6);
p.Color(4) = 0.15;
xlabel('time'); ylabel('prediction');
xlim([min(ts) max(ts)]); ylim([-9 14]);
legend('reference','KMD','location','northwest'); 
title('KMD prediction');
saveas(gcf,'KMD_inference','pdf');

%% set plotting preferences

function colors = set_plotting_preferences(iterset)

close all; 
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',20);
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',20);
set(groot,'DefaultTextFontSize',20);

k = length(iterset); colors = zeros(k,3);
for i=1:k
    colors(i,:) = (1-(i-1)/(k-1))*[0 1 1] ... 
                  + (i-1)/(k-1)*[1 0 1];   %color scheme
end

end
