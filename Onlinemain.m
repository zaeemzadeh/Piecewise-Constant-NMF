clear all
rng('default')
rng(2)

parameter = 0;%logspace(-3,-1,15);%0:0.1:0.5;%
MCtrials = 1;

MeanError = zeros(1,numel(parameter));
MeanOutlierNum = zeros(1,numel(parameter));
MeanTramsitionNum = zeros(1,numel(parameter));
for p = 1:numel(parameter)
%% Environment Parameters
    T = 3000;                        % number of samples
    DataDimension = 30;             
    ModelOrder = 3;                 % dimension of the activation matrix

    OutlierRatio = 0.3;
    MissingRatio = 0.3;

    NoiseVariance = 1e-2;            % additive Gaussian noise variance
    OutlierVariance = 1e-1;
    
    PowerAmp = 1;                 % outlier with uniform distribution
    TransitionRatio = 0.05;
    %% Algorithm Parameters
%      beta = 0.3664;                                % piecewise constant penalty regularization
    % calculating beta for normalized data S/PowerAmp, i.e. a = 1
    K = ModelOrder;                             % Model Order for algorithm
    a = 1; s = 1-TransitionRatio + TransitionRatio/(2*a); b = 1/log(2*a*s/(TransitionRatio));
    beta =  0.2*ones(K,1);%NoiseVariance/(b*PowerAmp.^2);%parameter(p);%
    lambda =  0;%0.2;% NoiseVariance/sqrt(parameter(p)/2); % 5e-1;                             % outlier sparsity penalty regularization
    tetha = 0.99;              %forgeting factor
%     MaxIter = 1e4;                              % maximum number of iterations
    
%     Delta = 1e-4;                               % convergence error
%     epsilon = 4*NoiseVariance/PowerAmp.^2; % IRLS parameter  : for the normalized data,i.e. S/PowerAmp 0.05 is a good choice


    error = zeros(1,MCtrials);
    NumofOutliershat = zeros(1,MCtrials);
    NumofTransitions = zeros(1,MCtrials);
    %% Monte Carlo trials
    for seed = 1:MCtrials
        rng(2);
        %% Initialization
        Gamma = rand(DataDimension,ModelOrder);    % signatures matrix
        % P = rand(ModelOrder,T);                    % activation matrix
         P = (2*PowerAmp*(rand(ModelOrder,T)-0.5)).*(rand(ModelOrder,T)<TransitionRatio); % Piecewise Constant activation matrix
         P(:,1) = PowerAmp*ones(ModelOrder,1);P = abs(cumsum(P,2));
%        P = abs(cumsum(laprnd(ModelOrder,T,0,0.1),2));
        W = rand(DataDimension,T) > MissingRatio;   % binary weight for the available data

        D = Gamma*P;                                % data matrix
%         O = (rand(size(D)) < OutlierRatio).*(2*PowerAmp*(rand(size(D))-0.5)); % outlier matrix
%        O = laprnd(DataDimension,T,0,parameter(p));
        O = (rand(size(D)) < OutlierRatio).*randn(DataDimension,T)*sqrt(OutlierVariance);
        N = randn(DataDimension,T)*sqrt(NoiseVariance); %  noise matrix

        S = D + O + N;                              % received (contaminated) data
        S(~W) = nan;                                % with missing data
        %% Factorization
        errorintime = zeros(1,T);
        Gammahat = rand(DataDimension,K);%Gamma;%
        pt = zeros(K,1);
        B = zeros(size(Gammahat));
        C = rand(K,K,DataDimension);
        Dhat = zeros(size(S));
        Phat = zeros(K,T);
        for t= 1:T
            beta = 2*sqrt(2 * NoiseVariance + 2*OutlierRatio*OutlierVariance) .* sqrt(sum(pinv(Gammahat).^2,2));
%             lambda =  2* sqrt(NoiseVariance);%0.2;% NoiseVariance/sqrt(parameter(p)/2); % 5e-1;      
            [Gammahat, pt, ot , B , C] = OnlineRobustPCNMF(S(:,t)./PowerAmp,W(:,t),Gammahat,B,C,pt,beta,lambda,tetha);
            Phat(:,t) = pt;

            if sum(pt < 1e-5)
                pt
            end
            
            window = max(t-500,1):t;            
%             if t >100
%                 tranRatio = sum(diff(Phat(:,window),1,2) > repmat(beta,1,numel(window)-1),2) / numel(window);
%                 beta = beta.*1.05.^(tranRatio > TransitionRatio);
%                 beta = beta.*0.95.^(tranRatio < TransitionRatio);
%             end
%             Gammahat = Gamma;
            pt = Phat(:,t);
            Dhat(:,t) = Gammahat*pt;
%             errorintime(t) = mean(norm(D(:,window) - Dhat(:,window))/norm(D(:,window)));
            errorintime(t) = mean(norm(D(:,t) - Dhat(:,t))/norm(D(:,t)));
        end
         error(seed) = rms(Dhat(:)-D(:))
%        error(seed) = rms(Dhat(W)-S(W));
%         NumofOutliershat(seed) = sum(sum(Ohat > 0.1 ))  ;
%         NumofTransitions(seed) = sum(sum(diff(Phat,1,2) > 0.1*PowerAmp )) ;
        [p/numel(parameter) seed/MCtrials]
    end
    MeanError(p) = median(error);
%     MeanOutlierNum(p) = mean(NumofOutliershat);
%     MeanTramsitionNum(p) = mean(NumofTransitions);
end
%% Plot Figures

hold on
 figure(6);clf
 plot(Phat(1,:))
 hold on
 plot(Phat(2,:))
 plot(Phat(3,:))
  plot(P(1,:),'-oy')
 plot(P(3,:),'-oy')
  plot(P(2,:),'-oy')
  %%
  figure(5)
semilogy(errorintime,'LineWidth',2,'DisplayName','Robust PC-NMF')
legend_handle  = legend('show');
set(legend_handle,'Interpreter','latex')
set(legend_handle,'FontSize',14)
set(legend_handle,'Location','Best')

yhandle = ylabel('RMSE');
set(yhandle,'Interpreter','latex','FontSize',16)

xhandle = xlabel('Time');
set(xhandle,'Interpreter','latex','FontSize',16)

set(gca,'FontSize',12,'LineWidth',1)
grid on

% % subplot(2,1,1)
%  figure(7);
% plot(parameter,MeanError,'-oy','LineWidth',4,'DisplayName','Robust PC-NMF')
% hold on
% % a = 1; s = 1-TransitionRatio + TransitionRatio/(2*a); b = 1/log(2*a*s/(TransitionRatio));betas = NoiseVariance/(b*PowerAmp.^2);
%  line([betas betas],[0 1],'LineWidth',4)
% 
% legend_handle  = legend('show');
% set(legend_handle,'Interpreter','latex')
% set(legend_handle,'FontSize',14)
% set(legend_handle,'Location','Best')
% 
% yhandle = ylabel('Averaged RMSE');
% set(yhandle,'Interpreter','latex','FontSize',16)
% 
% xhandle = xlabel('xxx');
% set(xhandle,'Interpreter','latex','FontSize',16)
% 
% set(gca,'FontSize',12,'LineWidth',1)
% grid on

% figure(1);
% subplot(2,1,2)
% plot(parameter,MeanTramsitionNum,'-or','LineWidth',4,'DisplayName','Robust PC-NMF')
% hold on
%  plot(parameter,MeanOutlierNum,'-og','LineWidth',4,'DisplayName','Robust PC-NMF')
% 
% legend_handle  = legend('show');
% set(legend_handle,'Interpreter','latex')
% set(legend_handle,'FontSize',14)
% set(legend_handle,'Location','Best')
% 
% yhandle = ylabel('Num of Outliers');
% set(yhandle,'Interpreter','latex','FontSize',16)
% 
% xhandle = xlabel('xxx');
% set(xhandle,'Interpreter','latex','FontSize',16)
% 
% set(gca,'FontSize',12,'LineWidth',1)
% grid on