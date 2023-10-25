function X = create_noise(d,thet,sig,dt)

%try X = create_noise(47,1,1,0.01);

%load data
load lorenz_data.mat X;

%introduce Ornstein-Uhlenbeck noise
[N,~] = size(X); noise = zeros(N,d);
for n=1:N-1
    noise(n+1,:) = ...
        noise(n,:)*(1-thet*dt) + sig*sqrt(dt)*normrnd(0,1,[1 d]);
end
X = [X,noise];

%save workspace
save noisy_lorenz_data.mat;
