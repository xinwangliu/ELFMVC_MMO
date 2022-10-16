function [Hstar,Sigma,obj] = minmaxLateFusionMVC(HP,WP,Sigma,option)

numker = size(HP,3);
numclass = size(HP,2);
num = size(HP,1);

% Sigma = ones(numker,1)/numker;

% KHP = zeros(num,num,numker);
% for p = 1:numker
%     KHP(:,:,p) = myLocalKernel(KH,tau,p);
% end
% KH = KHP;
% clear KHP
%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-2;
end
if ~isfield(option,'goldensearchmax')
    optiongoldensearchmax=1e-8;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end

nloop = 1;
loop = 1;
goldensearch_deltmaxinit = option.goldensearch_deltmax;
%%---
% MaxIter = 30;
% res_mean = zeros(4,MaxIter);
% res_std = zeros(4,MaxIter);
%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Hmatrix = zeros(num,numclass);
for p =1:numker
    Hmatrix = Hmatrix + Sigma(p)^2*HP(:,:,p)*WP(:,:,p);
end
[UH,SH,VH] = svd(Hmatrix,'econ');
Hstar = UH*VH';
obj(nloop) = sum(diag(SH));

[grad] = lateFusionMVCGrad(HP,Hstar,WP,Sigma);

Sigmaold  = Sigma;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%
while loop
    nloop = nloop+1;
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = lateFusionMVCupdate(HP,WP,Sigmaold,grad,obj(nloop-1),Hstar,option);

    % [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
    %     %-------------------------------
    %     % Numerical cleaning
    %     %-------------------------------
    %    Sigma(find(abs(Sigma<option.numericalprecision)))=0;
    %    Sigma = Sigma/sum(Sigma);
    
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision &&...
            option.goldensearch_deltmax > optiongoldensearchmax
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end
    
    [grad] = lateFusionMVCGrad(HP,Hstar,WP,Sigma);
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if  max(abs(Sigma-Sigmaold))<option.seuildiffsigma
        loop = 0;
        fprintf(1,'variation convergence criteria reached \n');
    end
    
    %     if nloop>=MaxIter
    %         loop = 0;
    %     end
    
    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------
    Sigmaold  = Sigma;
end