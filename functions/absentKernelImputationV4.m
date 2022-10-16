function [K]= absentKernelImputationV4(H,K_oo,mis_set)

%%the indices of missing samples.
alpha0 = 1e-4;
num = size(H,1);
n0 = length(mis_set);
Kx = H*H';
%% c is the indices of available samples.
obs_set = setdiff(1:num,mis_set);
%%
T_ou = Kx(obs_set,mis_set);
T_uu = Kx(mis_set,mis_set);
T_uu = (T_uu+T_uu')/2;
%% solving Wp iteratively
W = -T_ou/(T_uu + alpha0*eye(n0));
%% reconstructing Kp with Wp
K_ou = K_oo*W;
K_uu = W'*K_ou;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K(obs_set,obs_set) = K_oo;
K(obs_set,mis_set) = K_ou;
K(mis_set,obs_set) = K_ou';
K(mis_set,mis_set) = K_uu;
K = (K+K')/2;