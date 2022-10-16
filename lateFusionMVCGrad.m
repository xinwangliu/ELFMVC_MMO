function [grad] = lateFusionMVCGrad(HP,Hstar,WP,Sigma)

d=size(HP,3);
grad=zeros(d,1);
for p=1:d
     grad(p) = 2*Sigma(p)*trace(Hstar'*HP(:,:,p)*WP(:,:,p));  
end