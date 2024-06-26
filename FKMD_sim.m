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
h = input('bandwidth? ');
iters = input('number of iterations? ');

%set additional parameters as constant
sig = 1;          %Gaussian noise parameter
steps = 200;      %number of inference steps per iteration
samples = 5000;   %# of samples for computing Mahalanobis matrix
efcns = 20;       %# of Koopman eigenfunctions to keep
bta = 10^(-5);    %regularization parameter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% initialize simulation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load reference data
load lorenz96_data.mat ref;

%add noise and mean center data
ref = add_noise(ref,noise,sig);

%perform time embedding and generate validation data
[X,Y,d,obs_ref] = delay_embed(ref,steps,N,noise,l);

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
    [psi,dpsi] = get_fourier_features(X,M,R,d,h,N,samples);

    %do Koopman eigendecomposition
    [Psi_x,Psi_y] = get_feature_matrices(psi,X,Y);
    [K,Xi,Mu,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,Y,bta,R);

    %perform inference
    obs_inf = do_inference(Phi_x,V,Mu,steps,N,R,d,noise);

    %get mahalanobis matrix
    M = get_mahalanobis_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples);

    %save to workspace
    save(['FKMD_N',num2str(N), ...
        '_R',num2str(R), ...
        '_l',num2str(l), ...
        '_noise',num2str(noise), ...
        '_h',num2str(h), ...
        '_iter',num2str(iter), ...
        '_steps',num2str(steps)], ...
        "obs_inf","obs_ref","Mu","M","N","R","l","noise","h","steps","d");

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

function [X,Y,d,obs_ref] = delay_embed(ref,steps,N,noise,l)

disp('performing time embedding...')

%determine dimension of model
d = l*(noise+1);

%pull training data
X = zeros(N,d); Y = zeros(N,d);
for n=1:N
    X(n,:) = reshape(ref(n:n+l-1,:)',[1 d]); 
    Y(n,:) = reshape(ref(n+1:n+l,:)',[1 d]);
end

%pull validation data
obs_ref = ref(N+l:N+l+steps-1,:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [psi,dpsi] = get_fourier_features(X,M,R,d,h,N,samples)

disp('getting Fourier features...')

%get bandwidth
ind = randsample(N,samples,'true'); sig = median(pdist(X(ind,:)*M));

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

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,Y,bta,R)

disp('getting koopman modes...')

%get Koopman eigenfunctions
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*Y);

%get Koopman modes
V = B'*(W./diag(W'*Xi)');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function obs_inf = do_inference(Phi_x,V,Mu,steps,N,R,d,noise)

disp('doing inference...')

%initialize eigenvalues and inferred observations
D = ones(R,1); obs_inf = zeros(steps,noise+1);

%perform inference
for t = 1:steps

    %do Koopman inference
    o = real(Phi_x(N,:)*(D.*V')); 
    
    %pull last part of time embedding
    obs_inf(t,:) = o(d-noise:d);

    %update eigenvalues
    D = D.*Mu;

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = get_mahalanobis_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples)

disp('getting Mahalanobis matrix...')

%trim Koopman eigenfunctions and Koopman modes
Mu = Mu(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%define Jacobian function
XiLamV = Xi*(Mu.*V'); jacobian = @(x) dpsi(x)*XiLamV;

%initialize M and choose subsamples
M = zeros(d,d); ind = randi(N,[samples 1]);

%compute mahalanobis matrix M
for n=1:samples

    %compute curvature
    J = jacobian(X(ind(n),:)); 
    
    %update gradient outerproduct
    M = M + J*J';

end

%get square root of M (with optional normalization) 
M = real(sqrtm(real(M))); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%