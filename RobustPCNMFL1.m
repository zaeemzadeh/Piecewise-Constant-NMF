function [W, H, O] = RobustPCNMFL1(V,X,K,beta,lambda,maxIter,err,epsl)
% Robust Piecewise Constant Nonnegative Matrix Factorization using Ecleadian
% Distance as MEasure of Fit and IRLS for structure penalty
% V : Data
% X : Binary Weight for missing entries
% K : Number of Components
% beta : structure penalty 
% lambda: outlier sparsity penalty
% maxIter: maximum number of Iterations
% err: convergence criteria
% epsl: parameter in IRLS
% W: signatures
% H: Activation matrix
% O: Outlier matrix

%% initializing alg parameters
% X = double(~isnan(V));
V(isnan(V)) = eps;
V(V<=0) = eps;

W = rand(size(V,1),K);
H = rand(K,size(V,2));
O = rand(size(V));
Wk = zeros(size(W));
Hk = zeros(size(H));
Ok = rand(size(O));
%% iterations
for i = 1:maxIter
    if lambda == 0
        Vstar = V;
    else
        Vstar = V + O;
    end
    %% Update H
    lambdaH = ones(size(H));
    gradFH = -W'*(X.*Vstar - X.*(W*H) );
    KH  = W'*( X.*(W*H) );     % K * H   K defined in Algorithms for Non-negative Matrix Factorization
    
    b = zeros(size(H));
    y = 1./(abs(diff(H,1,2)) + epsl);
    
    b(:,1:end-1) = y.*H(:,2:end);
    b(:,2:end) = b(:,2:end) + y.*H(:,1:end-1);
    b =  2*beta* b.*lambdaH.^2;
    
    a = zeros(size(H));
    a(:,1:end-1) = y;
    a(:,2:end) = a(:,2:end) + y;
    a = a * 2 * beta.*lambdaH.^2;% + 2*gamma.*w.*lambdaH.^2 ; 
    
    if beta == 0
        H = H.*(-gradFH + KH )./(KH );
    else
        H = H.*(-gradFH + KH + b)./(KH + a.*H);
    end
    
    
    H(H<=0) = eps;
    %% Update W
    gradFW = -(X.*Vstar - X.*(W*H))*H';
    KW = (X.*(W*H))*H';            % K * W   K defined in Algorithms for Non-negative Matrix Factorization
    
    sk = sum (diff(H,[],2).^2./(diff(Hk,[],2).^2 + epsl),2)';
    sk = repmat(sk,size(W,1),1);            % structure penalty for each PU
    
    a = 0;
    sumwgk = 0;
    b = 2*beta*sk.*sumwgk;% + 2*gamma*Spk.*sumwgk;
    if beta == 0
        W =  W.*(-gradFW + KW)./(KW);    
    else
        W =  W.*(-gradFW + KW - b)./(KW + a.*W);
    end
    
    
     W(W<=0) = eps;
    %% Update O
    O = sign(W*H - V).*max(0,abs(W*H - V)-lambda);        %shrinkage
    
    %% Convergence check
    if (norm(W-Wk)/norm(W)< err || norm(H-Hk)/norm(H) < err )
        break;
    end
    Wk = W;
    Hk = H;
    
    normMatrix = diag(sqrt(sum(W.^2)));
    W = W/normMatrix;
    H = normMatrix * H;
%     figure(7)
%     subplot(3,2,5)
%     plot(H(1,:))
%     subplot(3,2,6)
%     plot(H(2,:))
%     pause(0.05)
end
normMatrix = diag(sqrt(sum(W.^2)));
W = W/normMatrix;
H = normMatrix * H;
end
