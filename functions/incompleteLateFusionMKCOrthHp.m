function [H_normalized,WP,HP,beta,obj] = incompleteLateFusionMKCOrthHp(KH,S,k,qnorm)

num = size(KH, 2); %the number of samples
numker = size(KH, 3); %m represents the number of kernels
maxIter = 100; %the number of iterations
[HP,WP] = myInitializationHp(KH,S,k);
HP00 = HP;
beta = ones(numker,1)*(1/numker)^(1/qnorm);

flag = 1;
iter = 0;
% res91 = zeros(maxIter+1,3,numker);
while flag
    iter = iter +1;
    %---the first step-- optimize H_star with given (HP, WP and beta)
    RpHpwp = zeros(num,k); % k - clusters, N - samples
    for p=1:numker
        RpHpwp = RpHpwp +  beta(p)*(HP(:,:,p)*WP(:,:,p));
    end
    [Uh,Sh,Vh] = svd(RpHpwp,'econ');
    Hstar = Uh*Vh';
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     res90(iter,:) = myNMIACC(Hstar,Y,k);
    
    %---the second step-- optimize WP with (HP, H_star and beta)
    WP = updateWPabsentClusteringV1(HP,Hstar);
    
    %---the third step-- optimize HP with (WP, H_star and beta)
    HP = updateHPabsentClusteringOrthHp(WP,Hstar,S,HP00);
    
    %---the fourth step-- optimize beta with (WP, H_star and HP)
    beta = updateBetaAbsentClustering(HP,WP,Hstar,qnorm);
    
    %---Calculate Obj--
    RpHpwp = zeros(num,k);
    for p = 1:numker
        RpHpwp = RpHpwp + beta(p)*HP(:,:,p)*WP(:,:,p);
    end
    obj(iter) = trace(Hstar'*RpHpwp);
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-4 || iter>maxIter)
        flag =0;
    end
%     if (iter>2) && (iter>maxIter)
%         flag =0;
%     end
end
H_normalized = Hstar./ repmat(sqrt(sum(Hstar.^2, 2)), 1,k);