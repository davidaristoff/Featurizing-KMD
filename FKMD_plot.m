%define parameters (from FKMD simulation)
dt = 0.05; iterset = [1 2 5];

%set plotting preferences
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',20);

%create plots
for i=1:length(iterset)

%get iteration number and load data
iter = iterset(i);
    load(['FKMD_N',num2str(N), ...
        '_R',num2str(R), ...
        '_l',num2str(l), ...
        '_noise',num2str(noise), ...
        '_h',num2str(h), ...
        '_iter',num2str(iter), ...
        '_steps',num2str(steps)], ...
        "obs_inf","obs_ref","M","d");

%plot inferences vs reference
figure('Position', [30 30 400 300]);
ts = (N+l)*dt:dt:(N+l+steps-1)*dt;
plot(ts,obs_ref(1:length(ts),1),'-.b','linewidth',2); hold on; 
plot(ts,obs_inf(1:length(ts),1),'-r','linewidth',2);
xlabel('$t$'); ylabel('$\theta_1(t)$');
xlim([min(ts) max(ts)]); ylim([-10 10]);
legend('reference','inferred','interpreter','latex');
title(['FKMD inference at iteration ',num2str(iter)]);
saveas(gcf,['inference_iter_',num2str(iter)],'epsc');

%get observation and nuisance indices
ind_o = logical(repmat([1,zeros(1,noise)],1,d/(noise+1))); 
ind_n = logical(1-ind_o);

%plot observation and nuisance mahalanobis matrices
figure('Position', [30 30 800 300]); 
subplot(1,2,1); imagesc(M(ind_o,ind_o)); colorbar; 
title(['non-nuisance \mbox{{\boldmath $M$}$^{1/2}$} after iteration ' ...
    ,num2str(iter)]);
axes('Position',[.085 .185 .25 .25]); box on;
M2 = M(ind_o,ind_o); imagesc(M2(end-14:end,end-14:end)); axis square;
subplot(1,2,2); imagesc(M(ind_n,ind_n)); colorbar; 
clim([min(M(ind_o,ind_o),[],'all') max(M(ind_o,ind_o),[],'all')]);
title(['nuisance \mbox{{\boldmath $M$}$^{1/2}$} after iteration ',num2str(iter)]);
saveas(gcf,['mahalanobis_iter_',num2str(iter)],'epsc');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%