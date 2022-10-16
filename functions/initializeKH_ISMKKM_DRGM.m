function KH = initializeKH_ISMKKM_DRGM(KH,WP,mis_set,obs_set)

numker = size(KH,3);
num = size(KH,1);
for p = 1 : numker
    mis_setp = mis_set{p};
    obs_setp = obs_set{p};
%     Wp = zeros(length(obs_set),length(mis_set));
    Wp = WP{p};
    % Wp =  [eye(length(mis_set));zeros(length(obs_set)-length(mis_set),length(mis_set))];
    KH(obs_setp,obs_setp,p) = KH(obs_setp,obs_setp,p);
    KH(obs_setp,mis_setp,p) = KH(obs_setp,obs_setp,p)*Wp;
    KH(mis_setp,obs_setp,p) = KH(obs_setp,mis_setp,p)';
    KH(mis_setp,mis_setp,p) = KH(mis_setp,obs_setp,p)*Wp;
%     KH(mis_setp,mis_setp,p) = eye(length(mis_setp));
end