% Copyright 2017 Alireza Zaeemzadeh
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Please cite the following reference if you use this code
% Missing spectrum-data recovery in cognitive radio networks using piecewise constant nonnegative matrix factorization
% A Zaeemzadeh, M Joneidi, B Shahrasbi, N Rahnavard
% Military Communications Conference, MILCOM 2015-2015 IEEE, 238-243
%
% Please report any bug at zaeemzadeh -at- knights -dot- ucf -dot- edu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, H] = nmfEUC(V,X,K,beta,maxIter,err,epsl)
% Piecewise Constant Nonnegative Matrix Factorization using Ecleadian
% Distance as Measure of Fit and IRLS for structure penalty
% V : Data
% X : Observed/Missing Data points
% K : Number of Components
% beta : structure penalty 
% maxIter: maximum number of Iterations
% epsl: parameter in IRLS
% W: signatures
% H: Activation matrix

%% initializing alg parameters
V(isnan(V)) = eps;
V(V<=0) = eps;

W = rand(size(V,1),K);
H = rand(K,size(V,2));
Wk = zeros(size(W));
Hk = zeros(size(H));

%% iterations
% update rules can be found in 
% Missing spectrum-data recovery in cognitive radio networks using piecewise constant nonnegative matrix factorization
% A Zaeemzadeh, M Joneidi, B Shahrasbi, N Rahnavard
% Military Communications Conference, MILCOM 2015-2015 IEEE, 238-243
for i = 1:maxIter
    %% update H
    lambdaH = ones(size(H));
    gradFH = -W'*(X.*V - X.*(W*H) );
    KH  = W'*( X.*(W*H) );     
    
    b = zeros(size(H));
    y = 1./(diff(H,1,2).^2 + epsl);
    
    b(:,1:end-1) = y.*H(:,2:end);
    b(:,2:end) = b(:,2:end) + y.*H(:,1:end-1);
    b =  2*beta* b.*lambdaH.^2;
    
    a = zeros(size(H));
    a(:,1:end-1) = y;
    a(:,2:end) = a(:,2:end) + y;
    a = a * 2 * beta.*lambdaH.^2;% 
    
    if beta == 0
        H = H.*(-gradFH + KH )./(KH );
    else
        H = H.*(-gradFH + KH + b)./(KH + a.*H);
    end
    
    
    H(H<=0) = eps;
    %% Update W
    gradFW = -(X.*V - X.*(W*H))*H';
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
    
    if (norm(W-Wk)/norm(W)< err || norm(H-Hk)/norm(H) < err )
        break;
    end
    Wk = W;
    Hk = H;
    

    lambda = diag(sqrt(sum(W.^2)));
    W = W/lambda;
    H = lambda * H;
end
lambda = diag(sqrt(sum(W.^2)));
W = W/lambda;
H = lambda * H;
end
