%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% FKMD with random Fourier features %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; set_plotting_preferences();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% define simulation parameters %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define parameters for double time embedded simulation
N = 10^5;         %number of samples for learning
R = 500;          %number of Fourier features
h = 1;            %bandwidth scaling factor
delay = 20;       %delay embedding length
noise = 2;        %number of noise coordinates
sig = 1;          %Gaussian noise parameter
steps = 100;      %number of inference steps per iteration
iters = 10;       %number of iterations
efcns = 50;       %# of Koopman eigenfunctions to keep
bta = 10^(-5);    %regularization parameter
samples = 5000;   %number of subsamples for getting bandwidth & M matrix

%set delay = 1, noise = 0, and iters = 1 to do ordinary KMD simulation
delay = 1; noise = 0; iters = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% initialize simulation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load and trim data
load lorenz96_data.mat ref;

%add noise and mean center data
ref = add_noise(ref,noise,sig);

%perform delay embedding
[X,Y,d] = delay_embed(ref,delay,N,noise);

%initialize Mahalanobis matrix and errors
M = eye(d); errs = zeros(iters,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('beginning simulation...')
tic

for iter = 1:iters
    disp(['beginning iteration # ...',num2str(iter)]);

    %update kernel function
    [z,dz,M] = get_fourier_features(X,M,R,N,d,h,samples);

    %do kernel DMD
    [Psi_x,Psi_y] = get_feature_matrices(z,X,Y);
    [K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R);

    %perform inference
    [obs_ref,obs_inf,errs] = ...
        do_inference(ref,Phi_x,V,Lam,steps,N,d,noise,delay,iter,errs);

    %plot results
    plot_results(obs_ref,obs_inf,steps,iter,M,N,d,delay,noise);

    %get mahalanobis matrix
    M = get_mahalanobis_matrix(dz,X,Xi,V,Lam,N,d,efcns,samples);

end

%plot errors
figure; plot(errs,'.b','markersize',20);

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% end simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ref = add_noise(ref,noise,sig)

%add noise
ref = [ref,normrnd(0,sig,[length(ref) noise])]; 

%mean-center data
ref = ref-mean(ref);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X,Y,d] = delay_embed(ref,delay,N,noise)

disp('performing time delay embedding...')

%determine dimension of model
d = delay*(noise+1);

%compute data matrices
X = zeros(N,d); Y = zeros(N,d);
for n=1:N
    X(n,:) = reshape(ref(n:delay+n-1,:)',[1 d]); 
    Y(n,:) = reshape(ref(n+1:delay+n,:)',[1 d]);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [z,dz,M] = get_fourier_features(X,M,R,N,d,h,samples)

disp('getting Fourier features...')

%update bandwidth by sampling pairwise standard deviations
ind = randi(N,[samples 1]); sig = std(pdist(X(ind,:)*M)); 

%get fourier features coefficients
w = normrnd(0,1,[d R]); M = M/(h*sig);

%define fourier features (matrix input)
z = @(X) sqrt(2/R)*exp(1i*X*M*w);

%define fourier features derivative function (row vector input)
dz = @(x) sqrt(2/R)*1i*exp(1i*x*M*w).*(M*w);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Psi_x,Psi_y] = get_feature_matrices(z,X,Y)

disp('getting feature matrices...')

%form data matrices
Psi_x = z(X); Psi_y = z(Y);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R)

disp('getting koopman singular vectors...')

%get estimated Koopman operator
K = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*Psi_y);

%get left and right eigenvectors of Koopman operator
[Xi,Lam,W] = eig(K,'vector');

%sort eigenvectors
[Lam,index] = sort(Lam,'descend'); Xi = Xi(:,index); W = W(:,index);

%create diagonal eigenvalue matrix
Lam = diag(Lam); 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R)

disp('getting koopman modes...')

%get Koopman eigenfunctions, Phi_x
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*X);

%get Koopman modes, V
V = B'*(W./diag(W'*Xi)');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obs_ref,obs_inf,errs] = ...
    do_inference(ref,Phi_x,V,Lam,steps,N,d,noise,delay,iter,errs)

disp('doing inference...')

%choose starting sample and compute reference time series
obs_ref = ref(delay+N:delay+N+steps-1,:);

%perform inference
D = Lam; obs_inf = zeros(steps,noise+1);
for step = 1:steps
    o = real(Phi_x(N,:)*D*V'); obs_inf(step,:) = o(d-noise:d); D = D*Lam;
end

%store errors
errs(iter) = norm(obs_ref-obs_inf,'fro')/steps;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_results(obs_ref,obs_inf,steps,iter,M,N,d,delay,noise)

disp('plotting results...'); 

%plot inferences vs reference
figure('Position', [30 30 400 300]); 
ts = (N+delay)*.06:0.06:(N+delay+steps-1)*0.06;
plot(ts,obs_ref(:,1),'-.b'); hold on; plot(ts,obs_inf(:,1),'-r');
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
title(['non-nuisance ${\bf M}$ after iteration ',num2str(iter-1)]);
subplot(1,2,2); imagesc(M(ind_n,ind_n)); colorbar; 
clim([min(M(ind_o,ind_o),[],'all') max(M(ind_o,ind_o),[],'all')]);
title(['nuisance ${\bf M}$ after iteration ',num2str(iter-1)]);
saveas(gcf,['mahalanobis_iter_',num2str(iter)],'epsc');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = get_mahalanobis_matrix(dz,X,Xi,V,Lam,N,d,efcns,samples)

disp('getting Mahalanobis matrix...')

%convert eigenvalue matrix to vector and compute log
LogLam = log(diag(Lam)).';

%trim Koopman eigenfunctions and Koopman modes
LogLam = LogLam(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%define Jacobian function
XiLogLamV = (Xi.*LogLam)*V'; jacobian = @(x) dz(x)*XiLogLamV;

%compute M as the outerproduct of curvature at subsamples
M = zeros(d,d); ind = randi(N,[samples 1]);
for i=1:samples
    J = jacobian(X(ind(i),:)); M = M + J*J';
end

%get square root and normalize
M = real(sqrtm(real(M))); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function set_plotting_preferences()

close all;
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',20);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%