%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% FKMD simulation of Lorenz 96 system%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define simulation parameters

%set data parameters
sig = 3;         %noise parameter
noise = 1;       %number of noise coordinates

%set main FKMD parameters
N = 1e6;         %number of samples
R = 5e3;         %number of features
iters = 5;       %number of iterations
l = 100;         %time delay length
h = 1;           %bandwidth

%set inference and other parameters
steps = 100;     %number of inference steps per iteration
samples = 5e3;   %# of samples for computing M matrix
efcns = 20;      %# of Koopman eigenfunctions to keep
bta = 1e-5;      %ridge regularization parameter

%% initialize simulations

%load reference data
load lorenz96_data.mat ref;

%add noise and mean center data
ref = add_noise(ref,noise,sig);

%perform time embedding; generate validation data
[X,Y,d,obs_ref] = delay_embed(ref,steps,N,noise,l);

%initialize M matrix
M = eye(d); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% begin simulations

disp('beginning simulation...')
tic

for iter = 1:iters
    disp(['beginning iteration # ...',num2str(iter)]);

    %update features
    [psi,dpsi] = get_fourier_features(X,M,R,d,h);

    %do Koopman eigendecomposition
    [Psi_x,Psi_y] = get_feature_matrices(psi,X,Y);
    [K,Xi,Mu,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R);
    [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R);

    %perform inference
    obs_inf = do_inference(Phi_x,V,Mu,steps,N,R,d);

    %get M matrix
    M = get_M_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples);

    %save to workspace
    save(['FKMD_iter',num2str(iter)], ...
        "obs_inf","obs_ref","Mu","M","N","R","l","noise","h","steps","d");

end

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% end simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% add noise

function ref = add_noise(ref,noise,sig)

%add noise
ref = [ref,normrnd(0,sig,[length(ref) noise])]; 

%mean-center data
ref = ref-mean(ref);

end

%% perform delay embedding

function [X,Y,d,obs_ref] = delay_embed(ref,steps,N,noise,l)

disp('performing time embedding...')

%determine dimension of model
d = l*(noise+1);

%initialize training and validation data
X = zeros(N+steps-1,d); Y = zeros(N+steps-1,d);

%define training and validation data
for n=1:N+steps-1
    X(n,:) = reshape(ref(n:n+l-1,:)',[1 d]); 
    Y(n,:) = reshape(ref(n+1:n+l,:)',[1 d]);
end

%define validation data
obs_ref = X(N:N+steps-1,:);

%trim data
X = X(1:N,:); Y = Y(1:N,:);

end

%% get random Fourier features

function [psi,dpsi] = get_fourier_features(X,M,R,d,h)

disp('getting Fourier features...')

%get bandwidth
sig = sqrt(sum(var(X*M)));

%get fourier features coefficients
w = normrnd(0,1,[d R]); M = M/(h*sig);

%define fourier features (matrix input)
psi = @(X) sqrt(2/R)*exp(1i*X*M*w);

%define fourier features derivative function (row vector input)
dpsi = @(x) sqrt(2/R)*1i*exp(1i*x*M*w).*(M*w);

end

%% get feature matrices

function [Psi_x,Psi_y] = get_feature_matrices(psi,X,Y)

disp('getting feature matrices...')

%form data matrices
Psi_x = psi(X); Psi_y = psi(Y);

end

%% get Koopman eigenvectors

function [K,Xi,Mu,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,R)

disp('getting koopman eigenvectors...')

%get Koopman matrix
K = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*Psi_y);

%eigendecompose Koopman operator
[Xi,Mu,W] = eig(K,'vector');

%sort eigenvectors
[Mu,index] = sort(Mu,'descend'); Xi = Xi(:,index); W = W(:,index);

end

%% get koopman modes

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R)

disp('getting koopman modes...')

%get Koopman eigenfunctions
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*X);

%get Koopman modes
V = B'*(W./diag(W'*Xi)');

end

%% do inference

function obs_inf = do_inference(Phi_x,V,Mu,steps,N,R,d)

disp('doing inference...')

%initialize eigenvalues and inferred observations
D = ones(R,1); obs_inf = zeros(steps,d);

%perform inference
for t = 1:steps

    %do Koopman inference
    obs_inf(t,:) = real(Phi_x(N,:)*(D.*V'));

    %update eigenvalues
    D = D.*Mu;

end

end

%% get M matrix

function M = get_M_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples)

disp('getting M matrix...')

%trim Koopman eigenfunctions and Koopman modes
Mu = Mu(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%define Jacobian function
XiLamV = Xi*((Mu-1).*V'); jacobian = @(x) dpsi(x)*XiLamV;

%initialize M and choose subsamples
M = zeros(d,d); ind = randi(N,[samples 1]);

%compute M matrix
for n=1:samples

    %compute curvature
    J = jacobian(X(ind(n),:)); 
    
    %update gradient outerproduct
    M = M + J*J';

end

%get square root of M and normalize 
M = real(sqrtm(real(M))); M = M/norm(M);

end
