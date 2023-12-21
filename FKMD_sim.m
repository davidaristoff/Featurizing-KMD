%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% FKMD with random Fourier features %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%runs FKMD as described in Algorithm II.1 of arXiv:2312.09146v2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% define simulation parameters %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose parameters for FKMD simulation
N = input('number of samples? ');
R = input('number of features? ');
l = input('time delay length? ');
noise = input('number of noise coordinates? ');
iters = input('number of iterations? ');

%set additional parameters as constant
h = 1;            %bandwidth scaling factor
sig = 1;          %Gaussian noise parameter
steps = 200;      %number of inference steps per iteration
efcns = 20;       %# of Koopman eigenfunctions to keep
bta = 10^(-5);    %regularization parameter
samples = 5000;   %number of subsamples for getting bandwidth & M matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% initialize simulation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load reference data
load lorenz96_data.mat ref;

%add noise and mean center data
ref = add_noise(ref,noise,sig);

%perform l embedding
[X,Y,d] = l_embed(ref,N,noise,l);

%initialize Mahalanobis matrix
M = eye(d); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('beginning simulation...')
tic

for iter = 1:iters
    disp(['beginning iteration # ...',num2str(iter)]);

    %update features
    [psi,dpsi] = get_fourier_features(X,M,R,N,d,h,samples);

    %do Koopman eigendecomposition
    [Psi_x,Psi_y] = get_feature_matrices(psi,X,Y);
    [K,Xi,Mu,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R);

    %perform inference
    [obs_ref,obs_inf] = ...
        do_inference(ref,Phi_x,V,Mu,steps,N,R,d,noise,l);

    %get mahalanobis matrix
    M = get_mahalanobis_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples);

    %save to workspace
    save(['FKMD_N',num2str(N), ...
        '_R',num2str(R), ...
        '_l',num2str(l), ...
        '_noise',num2str(noise), ...
        '_iter',num2str(iter)], ...
        "obs_inf","obs_ref","M","N","R","l","noise","steps","d");

end

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

function [X,Y,d] = l_embed(ref,N,noise,l)

disp('performing time embedding...')

%determine dimension of model
d = l*(noise+1);

%compute data matrices
X = zeros(N,d); Y = zeros(N,d);
for n=1:N
    X(n,:) = reshape(ref(n:l+n-1,:)',[1 d]); 
    Y(n,:) = reshape(ref(n+1:l+n,:)',[1 d]);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [psi,dpsi] = get_fourier_features(X,M,R,N,d,h,samples)

disp('getting Fourier features...')

%update bandwidth by sampling pairwise standard deviations
ind = randi(N,[samples 1]); sig = std(pdist(X(ind,:)*M)); 

%get fourier features coefficients
w = normrnd(0,1,[d R]); M = M/(h*sig);

%define fourier features (matrix input)
psi = @(X) sqrt(2/R)*exp(1i*X*M*w);

%define fourier features derivative function (row vector input)
dpsi = @(x) sqrt(2/R)*1i*exp(1i*x*M*w).*(M*w);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Psi_x,Psi_y] = get_feature_matrices(psi,X,Y)

disp('getting feature matrices...')

%form data matrices
Psi_x = psi(X); Psi_y = psi(Y);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [K,Xi,Mu,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R)

disp('getting koopman eigenvectors...')

%get Koopman matrix
K = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*Psi_y);

%eigendecompose Koopman operator
[Xi,Mu,W] = eig(K,'vector');

%sort eigenvectors
[Mu,index] = sort(Mu,'descend'); Xi = Xi(:,index); W = W(:,index);

%create diagonal eigenvalue matrix
Mu = diag(Mu); 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R)

disp('getting koopman modes...')

%get Koopman eigenfunctions
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*X);

%get Koopman modes, V
V = B'*(W./diag(W'*Xi)');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obs_ref,obs_inf] = ...
    do_inference(ref,Phi_x,V,Mu,steps,N,R,d,noise,l)

disp('doing inference...')

%choose starting sample and compute reference time series
obs_ref = ref(l+N-1:l+N+steps-2,:);

%perform inference
D = eye(R); obs_inf = zeros(steps,noise+1);
for step = 1:steps
    o = real(Phi_x(N,:)*D*V'); obs_inf(step,:) = o(d-noise:d); D = D*Mu;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = get_mahalanobis_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples)

disp('getting Mahalanobis matrix...')

%convert eigenvalue matrix to vector and compute log
Lam = log(diag(Mu)).';

%trim Koopman eigenfunctions and Koopman modes
Lam = Lam(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%define Jacobian function
XiLamV = (Xi.*Lam)*V'; jacobian = @(x) dpsi(x)*XiLamV;

%compute M as the outerproduct of curvature at subsamples
M = zeros(d,d); ind = randi(N,[samples 1]);
for i=1:samples
    J = jacobian(X(ind(i),:)); M = M + J*J';
end

%get square root and normalize
M = real(sqrtm(real(M))); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%