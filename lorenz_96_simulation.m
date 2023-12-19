%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% simulate lorenz 96 system %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; figure;

%set parameters
d = 40;          %dimension, or number of spatial grid points
F = 8;           %forcing term
dt = 0.01;       %time step
T = 5*10^7;      %number of samples

%initialize solution array
X = zeros(T,d);

%set initial conditions
x = F + (mod((1:1:d)',5)==0);

%simulate using RK4
for t = 1:T
    k1 = dt*lorenz_96(x, d, F);
    k2 = dt*lorenz_96(x + 0.5*k1, d, F);
    k3 = dt*lorenz_96(x + 0.5*k2, d, F);
    k4 = dt*lorenz_96(x + k3, d, F);
    x = x + (k1 + 2*k2 + 2*k3 + k4)/6;
    X(t,:) = x';
end

%plot results if desired
theta = 0:2*pi/d:2*pi;
for t=1:T
    polarplot(theta,[X(t,:),X(t,1)],'color','m'); 
    rlim([-10 10]); pause(10^(-5));
end

function dxdt = lorenz_96(x,d,F)
    dxdt = zeros(d, 1);
    for i = 1:d
        im1 = mod(i-2+d,d)+1;   %i-1 in the cyclic sense
        im2 = mod(i-3+d,d)+1;   %i-2 in the cyclic sense
        ip1 = mod(i,d)+1;       %i+1 in the cyclic sense       
        dxdt(i) = (x(ip1)-x(im2))*x(im1)-x(i)+F;
    end
end