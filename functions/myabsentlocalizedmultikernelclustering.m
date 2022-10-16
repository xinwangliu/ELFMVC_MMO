function [H_normalized,gamma,obj,KA] = myabsentlocalizedmultikernelclustering(K,S,cluster_count,qnorm,lambda,numSel,algorithm_choose)

num = size(K,1);
nbkernel = size(K,3);
alpha0 = 1e-4;
%% S: num0*m, each column indicates the indices of absent samples
%% initialize kernel weights
gamma = ones(nbkernel,1)/nbkernel;
%% initialize base kernels with zeros
if strcmp(algorithm_choose,'algorithm0')
    KA = feval(algorithm_choose,K,S,7);
else
    KA = feval(algorithm_choose,K,S);
end
%% combining the base kernels
KC  = mycombFun(KA,gamma.^qnorm);
%%Calculate Neighborhood of each sample
NS = genarateNeighborhood(KC,numSel);%% tau*num
HE0 = calHessian(KA,NS,0);
%%--Calculate Neighborhood--%%%%%%
A0 = zeros(num);
for i =1:num
    A0(NS(:,i),NS(:,i)) = A0(NS(:,i),NS(:,i))+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag = 1;
iter = 0;
while flag
    iter = iter + 1;
    %% update H with KC
%     fprintf(1, 'running iteration of the proposed algorithm %d...\n', iter);
    H = mylocalkernelkmeans(KC,A0,cluster_count);
%     %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     res10(iter,:) = myNMIACC(H,Y,cluster_count);
   %% updata base kernels
    IH = eye(num) - H*H';
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Kx = zeros(num);
%     for i =1:num
%         A = zeros(num);
%         A(NS(:,i),NS(:,i)) = IH(NS(:,i),NS(:,i));
%         Kx = Kx + (1/num)*A;
%     end
    Kx = (1/num)*(A0.*IH);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    KA = zeros(num,num,nbkernel);
    for p =1:nbkernel
        if isempty(S{p}.indx)
            KA(:,:,p) = K(:,:,p);
        else
            mis_indx = S{p}.indx;
            obs_indx = setdiff(1:num,mis_indx);
            KA(:,:,p) = absentKernelImputation(Kx,K(obs_indx,obs_indx,p),mis_indx,alpha0);
        end
    end
    %% update kernel weights
    ZH = callZH(KA,Kx);
    obj(iter)  = callocalObj(HE0,ZH,gamma,lambda);
    [gamma]= updatelocalkernelweights(HE0,ZH,lambda);
    % gamma = (1./ZH)/sum((1./ZH));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% KC  = mycombFun(KA,gamma.^qnorm);
    if iter>2 && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 ||iter>100)
        flag =0;
    end
%     if iter>2 && (iter>50)
%         flag =0;
%     end
    KC  = mycombFun(KA,gamma.^qnorm);
    % A = genarateNeighborhood(KC,numsel0);
end
H_normalized = H./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);