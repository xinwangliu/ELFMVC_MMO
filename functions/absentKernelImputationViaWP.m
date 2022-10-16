function [K]= absentKernelImputationViaWP(H,K_oo,mis_set,alpha0)

%%the indices of missing samples.
num = size(H,1);
n0 = length(mis_set);
Kx = eye(num)-H*H';
%% c is the indices of available samples.
obs_set = setdiff(1:num,mis_set);
%%
T_ou = Kx(obs_set,mis_set);
T_uu = Kx(mis_set,mis_set);
T_uu = (T_uu+T_uu')/2;
W = -T_ou/(T_uu + alpha0*eye(n0));
K_ou = K_oo*W;
K_uu = W'*K_ou;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K(obs_set,obs_set) = K_oo;
K(obs_set,mis_set) = K_ou;
K(mis_set,obs_set) = K_ou';
K(mis_set,mis_set) = K_uu;
K = (K+K')/2;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lxcmmm = -Lxcm/(Lxmm + alpha0*eye(n0));
% Kycm = Kycc*LxcWmmm;
% Kymm = Lxcmmm'*Kycm;
% Kyr0 = [Kycc Kycm; Kycm' Kymm];
% Kyr0 = (Kyr0+Kyr0')/2;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [val,indxxx] = sort([cset,mset],'ascend');
% Kyr = Kyr0(indxxx,indxxx);
% % Kyr = Kyr/trace(Kyr);