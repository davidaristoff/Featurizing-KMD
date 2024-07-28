%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% basic FKMD simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% define simulation parameters %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose basic parameters for FKMD simulation
R = 1e3;          %number of features
h = 1;            %bandwidth scaling factor
efcns = 5e2;      %number of eigenfunctions (defines mode set S)
iters = 3;        %number of iterations

%choose other parameters for inference
steps = 1e3;     %number of inference steps per iteration
samples = 5e3;   %number of subsamples for getting M matrix
bta = 1e-5;      %regularization parameter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% initialize simulation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load input and output data (X = input data matrix, Y = output data matrix)
load system_data.mat X Y; 

%get data dimensions (N = number of samples, d = dimension of samples)
[N,d] = size(X);

%the required data is:
%X = input data matrix, an array of size Nxd
%Y = output data matrix, an array of size Nxd

%X and Y are simply input/output data pairs. Y(n,:) is the system
%state at lag tau, given that it started at X(n,:). System states need 
%not be delay embedded. If they are not, d is just the system's dimension. 
%With delay embedding, d = (system's dimension)x(l = embedding length).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%FKMD produces an array "obs_inf" at each iteration. 
%"obs_inf" predicts the system's evolution starting from X(N,:).
%the first dimension of "obs_inf" is time, so obs_inf(1,:) = X(N,:).

disp('beginning simulation...')
tic

%initialize Mahalanobis matrix
M = eye(d); 

%begin FKMD iterations
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
        "obs_inf","Mu","M","efcns","N","R","steps","d");

end

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% end simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [psi,dpsi] = get_fourier_features(X,M,R,d,h)

disp('getting Fourier features...')

%update bandwidth by sampling pairwise standard deviations
sig = sqrt(sum(var(X)));

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

function [Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,bta,R)

disp('getting koopman modes...')

%get Koopman eigenfunctions
Phi_x = Psi_x*Xi;

%get coordinates of observations
B = (Psi_x'*Psi_x+bta*eye(R))\(Psi_x'*X);

%get Koopman modes
V = B'*(W./diag(W'*Xi)');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = get_M_matrix(dpsi,X,Xi,V,Mu,N,d,efcns,samples)

disp('getting M matrix...')

%trim Koopman eigenfunctions and Koopman modes
Mu = Mu(1:efcns); Xi = Xi(:,1:efcns); V = V(:,1:efcns);

%define Jacobian function
XiLamV = Xi*((Mu-1).*V'); jacobian = @(x) dpsi(x)*XiLamV;

%initialize M and choose subsamples
M = zeros(d,d); ind = randi(N,[samples 1]);

%compute M
for n=1:samples

    %compute J
    J = jacobian(X(ind(n),:)); 
    
    %update M
    M = M + J*J';

end

%get square root of M (with optional normalization) 
M = real(sqrtm(real(M))); M = M/norm(M);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
