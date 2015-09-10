function [W, h, o , B , D] = OnlineRobustPCNMF(v,x,W,B,D,ht,beta,lambda,tetha)
% Robust Piecewise Constant Nonnegative Matrix Factorization using Ecleadian
% Distance as MEasure of Fit and IRLS for structure penalty
% v : New Data
% x : Binary Weight for missing entries
% W : signatures
% ht: Previous activation vector
% beta : structure penalty 
% lambda: outlier sparsity penalty
% tetha: forgetting factor
% maxIter: maximum number of Iterations
% epsl: parameter in IRLS
% W: signatures
% h: New Activation vector
% o: New Outlier vector

%% initializing alg parameters
% X = double(~isnan(V));
v(isnan(v)) = eps;
v(v<=0) = eps;

%% Update h
% h = W(x,:)\v(x,:);
h = lsqnonneg(W(x,:),v(x,:));
hdiff = h - ht;

 hdiff = sign(hdiff).*max(0,abs(hdiff)-beta);
%hdiff = (abs(hdiff)> beta).*hdiff;
h = ht + hdiff;
%% Update o
% o = x.*(W*h - v).*(abs(W*h - v)>lambda);
o = x.*sign(W*h - v).*max(abs(W*h - v),lambda);
%% Update W
% B = tetha* B + (x.*v)*h';
if lambda
   vstar = v  + o;
else
   vstar = v;
end

for p = 1:numel(v)
    B(p,:) = tetha* B(p,:) + x(p)*vstar(p)*h';
%     D(:,:,p) = D(:,:,p)/tetha - x(p)*D(:,:,p)*(h*h')*D(:,:,p)/(1 + h'*D(:,:,p)*h);
    D(:,:,p) = D(:,:,p)*tetha + x(p)*(h*h');%D(D<0) = 0;
    W(p,:) = lsqnonneg(D(:,:,p),B(p,:)');
 end
%  W(W<0) = 0;

end
