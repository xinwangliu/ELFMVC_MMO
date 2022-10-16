function [Wp,obj] = solvingWpV2(T_ou,T_uu,K_oo,Wp0)

flag =1;
iter = 0;
while flag
    iter = iter+1;
    M = 2*K_oo*(Wp0*T_uu + T_ou);
    [U,S,V] = svd(M,'econ');
    Wp = U*V';
    obj(iter) = trace(Wp'*K_oo*(2*T_ou+Wp*T_uu));
    if(iter>2&& ((obj(iter)-obj(iter-1))/obj(iter))<1e-4)
        flag = 0;
    else
        Wp0 = Wp;
    end
end