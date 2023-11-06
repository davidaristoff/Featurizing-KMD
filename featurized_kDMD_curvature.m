%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% time embedded kernel DMD with auto feature-finder %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% curvature-based Mahalanobis matrix, via equation (0.9) in the notes %%%

close all; set_plotting_preferences();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% initialize simulation parameters %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load data, find its dimension, and store as reference
load noisy_lorenz_data.mat X;

%get size of X and store it as reference dataset
[~,d] = size(X); Xref = X;

%use only part of the data for training
N = 10; X = X(1:N,:);

%define bandwidth, # of inference steps, Mahalanobis matrix, and observable
s = 0.05;        %bandwidth scaling factor
steps = 10;      %number of inference steps per iteration
iters = 2;       %number of iterations
efcns = 5;      %# of Koopman eigenfunctions to use for Mahalanobis matrix
bta = 10^(-2);   %regularization parameter
M = eye(d);      %initial (square root of) Mahalanobis matrix
obs = @(x) x;    %observable of interest

%initialize matrix of correlations
corrs = zeros(steps,iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('beginning simulation...')
tic

for iter = 1:iters
    disp(['beginning iteration # ...',num2str(iter)]);

    %update kernel function
    k = get_kernel_function(X,M,N,s);

    %do kernel DMD
    [Psi_x,Psi_y] = get_kernel_matrices(k,X,N);
    [K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,N);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,obs,N);

    %perform inference
    [obs_ref,obs_inf] = do_inference(Xref,Phi_x,V,Lam,obs,N,steps,d);
    corrs(:,iter) = plot_results(obs_ref,obs_inf,steps,M);

    %get mahalanobis matrix
    M = get_mahalanobis_matrix(k,X,Xi,V,Lam,M,N,d,efcns);

end

disp('mean correlations by iteration count...')
mean(corrs)

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function k = get_kernel_function(X,M,N,s)

%trim X
X = X(1:N-1,:);

%update bandwidth
h = s*std(pdist(sqrt(real(X*M).^2+imag(X*M).^2)))^2;

%update kernel function
k = @(Y) exp((-pdist2(real(Y*M),real(X*M)).^2 ... 
              -pdist2(imag(Y*M),imag(X*M)).^2)/(2*h)); 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Psi_x,Psi_y] = get_kernel_matrices(k,X,N)

disp('getting kernel matrices...')

%form kernel matrices
Psi_x = k(X(1:N-1,:)); Psi_y = k(X(2:N,:));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,N)

disp('getting koopman singular vectors...')

%get estimated Koopman operator
K = (Psi_x+bta*eye(N-1))\Psi_y;

%get left and right eigenvectors of Koopman operator
[Xi,Lam,W] = eig(K,'vector');

%sort eigenvectors
[Lam,index] = sort(Lam,'descend'); Xi = Xi(:,index); W = W(:,index);

%create diagonal eigenvalue matrix
Lam = diag(Lam); 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,obs,N)

disp('getting koopman modes...')

%get Koopman eigenfunctions, Phi_x
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = pinv(Psi_x)*obs(X(1:N-1,:));

%get Koopman modes, V
V = B'*(W./diag(W'*Xi)');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obs_ref,obs_inf] = do_inference(Xref,Phi_x,V,Lam,obs,N,steps,d)

disp('doing inference...')

%choose start time for inference
start = N-1;

%compute reference observable time series
obs_ref = obs(Xref(start+1:start+steps,:));

%perform inference
D = Lam; obs_inf = zeros(steps,d);
for step = 1:steps
    obs_inf(step,:) = real(Phi_x(start,:)*D*V'); D = D*Lam;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function corrs = plot_results(obs_ref,obs_inf,steps,M)

disp('plotting results...'); 

%open and place figure
figure('Position', [30 30 1400 1100]);

%compute correlations
[corrs,~] = corrcoef([obs_inf',obs_ref']); 

%trim correlations
corrs = diag(corrs(1:steps,steps+1:2*steps));

%plot correlations
subplot(2,2,1); 
plot(corrs); title('inference correlation function');
xlabel('time step'); ylabel('correlation');

%plot inferences vs reference
subplot(2,2,2); 
plot3(obs_ref(:,1),obs_ref(:,2),obs_ref(:,3),'ob');
hold on; plot3(obs_inf(:,1),obs_inf(:,2),obs_inf(:,3),'xr'); 
legend('reference','inferred','interpreter','latex');
title('trajectory inference');

%plot (square root of) mahalanobis matrix
subplot(2,2,3);
imagesc(real(M)); title('Mahalanobis matrix (real part)');

%plot (square root of) mahalanobis matrix
subplot(2,2,4);
imagesc(imag(M)); title('Mahalanobis matrix (complex part)');

%pause briefly
%pause(0.1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = get_mahalanobis_matrix(k,X,Xi,V,Lam,M,N,d,efcns)

disp('getting Mahalanobis matrix...')

%convert eigenvalue matrix to vector and compute log
LogLam = log(diag(Lam)).';

%trim Koopman eigenfunctions and Koopman modes
LogLam = LogLam(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%precompute some matrices
M2 = M*M'; X = X(1:N-1,:); XiLogLamV = (Xi.*LogLam)*V';

%define Jacobian function
jacobian = @(x) (M2*(X-x)'.*k(x))*XiLogLamV;

%compute M as the outerproduct
M = zeros(d,d);
for n=1:N-1
    J = jacobian(X(n,:)); M = M + J*J';
end

%get square root and regularize M
M = sqrtm(M); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function set_plotting_preferences()

close all;
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',14);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%