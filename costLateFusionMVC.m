function [cost,Hstar] = costLateFusionMVC(HP,WP,StepSigma,DirSigma,Sigma)

global nbcall
nbcall=nbcall+1;

Sigma = Sigma+ StepSigma * DirSigma;

num = size(HP,1);
numclass = size(HP,2);
numker = size(HP,3);

Hmatrix = zeros(num,numclass);
for p =1:numker
    Hmatrix = Hmatrix + Sigma(p)^2*HP(:,:,p)*WP(:,:,p);
end
[UH,SH,VH] = svd(Hmatrix,'econ');
Hstar = UH*VH';
cost = sum(diag(SH));