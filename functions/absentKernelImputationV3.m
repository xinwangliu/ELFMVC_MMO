function [K,obj0]= absentKernelImputationV3(H,K_oo,mis_set)

%%the indices of missing samples.
num = size(H,1);
IL = H*H';
%% c is the indices of available samples.
obs_set = setdiff(1:num,mis_set);
%%
T_ou = IL(obs_set,mis_set);
T_uu = IL(mis_set,mis_set);
T_uu = (T_uu+T_uu')/2;
%% Initialization
[U0,S0,V0] = svd(T_ou,'econ');
Wp0 = U0*V0';
% Wp0 = orth(T_ou);
%% solving Wp iteratively
[W,obj0] = solvingWpV2(T_ou,T_uu,K_oo,Wp0);
%% reconstructing Kp with Wp
K_ou = K_oo*W;
K_uu = W'*K_ou;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K(obs_set,obs_set) = K_oo;
K(obs_set,mis_set) = K_ou;
K(mis_set,obs_set) = K_ou';
K(mis_set,mis_set) = K_uu;
K = (K+K')/2;