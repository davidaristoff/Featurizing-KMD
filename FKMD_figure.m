%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% time embedded kernel DMD with auto feature-finder %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; set_plotting_preferences();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% initialize simulation parameters %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load data, find its dimension, and store as reference
load noisy_lorenz_data.mat X;

%do delay embedding of length embed_length
delay = 2; X = delay_embed(X,delay);

%get size of X and store it as reference dataset
[N,d] = size(X); Xref = X;

%use only part of the data for training
N = floor(N/2); X = X(1:N,:);

%define bandwidth, # of inference steps, Mahalanobis matrix, and observable
h = .05;          %bandwidth scaling factor
steps = 40;      %number of inference steps per iteration
iters = 3;       %number of iterations
efcns = 100;     %# of Koopman eigenfunctions to keep
bta = 10^(-5);   %regularization parameter
M = eye(d);      %initial **square root of** Mahalanobis matrix
obs = @(x) x;    %observable of interest

%initialize matrix of correlations
corrs = zeros(steps,iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('beginning simulation...')

%store inference at each iteration
obsinf = zeros(steps,d,iters);

for iter = 1:iters
    disp(['beginning iteration # ...',num2str(iter)]);

    %update kernel function
    k = get_kernel_function(X,M,N,h);

    %do kernel DMD
    [Psi_x,Psi_y] = get_kernel_matrices(k,X,N);
    [K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,N);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,obs,bta,N);

    %perform inference
    [obs_ref,obs_inf] = do_inference(Xref,Phi_x,V,Lam,obs,N,steps,d);
    corrs(:,iter) = get_corrs(obs_ref,obs_inf,steps);

    %get **square root of** mahalanobis matrix
    M = get_mahalanobis_matrix(k,X,Xi,V,Lam,M,N,d,efcns);

    %save inference
    obsinf(:,:,iter) = obs_inf;

end

%plot correlations at first step
figure; subplot(1,2,1); plot(corrs(end,:)); 
xlabel('iteration'); ylabel('correlation'); ylim([-1.1 1.1]);
title('correlation by iteration');

%plot inference at final step
subplot(1,2,2); ts = 1:1:steps;
plot(ts,obsinf(:,1,iters),'ob',ts,obs_ref(:,1),'-.b'); hold on; 
plot(ts,obsinf(:,2,iters),'or',ts,obs_ref(:,2),'-.r'); 
plot(ts,obsinf(:,3,iters),'og',ts,obs_ref(:,3),'-.g');
xlabel('time'); ylabel('system state');
title('inference after 3rd iteration');
legend('inferred','reference');

figure; M2 = M^2; sig = std(pdist(X*M)); imagesc(M2/(h*sig)^2); colorbar;
title('Mahalanobis matrix'); 
axes('Position',[.6 .7 .2 .2]); box on;
imagesc(M2(51:60,51:60)/(h*sig)^2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% end simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X = delay_embed(data,delay)

disp('performing time delay embedding...')

[N,L] = size(data); X = zeros(N-delay,delay*L);
for n=1:N-delay+1
    X(n,:) = reshape(data(n:n+delay-1,:)',[1 delay*L]);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function k = get_kernel_function(X,M,N,h)

%trim X
X = X(1:N-1,:);

%update bandwidth
sig = std(pdist(X*M));

%update kernel function
k = @(Y) exp(-pdist2(Y*M,X*M).^2/(h*sig)^2);  

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

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,obs,bta,N)

disp('getting koopman modes...')

%get Koopman eigenfunctions, Phi_x
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x+bta*eye(N-1))\obs(X(1:N-1,:));

%get Koopman modes, V
V = B'*(W./diag(W'*Xi)');

%get Koopman modes by ridge regression
%Note: In practice, use a built-in, NOT the normal equations!
%thta = 10^(-8); V = ((Xi'*Xi+thta*eye(N-1))\(Xi'*B))';

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

function corrs = get_corrs(obs_ref,obs_inf,steps)

disp('getting correlations...'); 

%compute correlations
[corrs,~] = corrcoef([obs_inf(:,1:3)',obs_ref(:,1:3)']); 

%trim correlations
corrs = diag(corrs(1:steps,steps+1:2*steps));

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

%get square root M and normalize
M = real(sqrtm(real(M))); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function set_plotting_preferences()

close all;
%set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',14);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%