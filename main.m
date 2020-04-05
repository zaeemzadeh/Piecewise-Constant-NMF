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
clear all
rng('default')
%%
NumberOfPUs = 2;
NumberOfSUs = 10; %;       %20
gridsize =50;
resol = 0.2;

ActivityChance = 0.7;
NumberOfActiveSUs = ceil(NumberOfSUs*ActivityChance);          %active SUs in each time slot
NumberOfInactiveSUs = NumberOfSUs - NumberOfActiveSUs;

alpha = 2.5;            %pathloss model
noise = 10.^(-40/10);        %variance of additive noise -60 dBm
shadowing = 0.5;%params(param); % ind dBm
Pmin = 20;  % dBm
Pmax = 27; % dBm

utilization = 0.3;          % channel usage percentage
Persistence = 0.95 + rand(1,NumberOfPUs)*0.0;        % active to active (PU)
Activation = (1-Persistence)*utilization/(1-utilization);         % silent to active (PU)

Window = 100; %25*NumberOfSUs*NumberOfPUs;
simTime = 1*Window ;

frequency = 2.4e6;
c = 3e8;

PL0 = 1;%(c/(4*pi*frequency))^-alpha;
d0 = 1;
%%
Pinput = zeros(NumberOfSUs,simTime);
 Pshadow = zeros(NumberOfSUs,simTime);
Poutput = zeros(NumberOfSUs,simTime);
KnownData = zeros(NumberOfSUs,simTime);
residuals = zeros(NumberOfSUs,simTime);
labels = zeros(1,simTime);
states = nan(NumberOfPUs,simTime);
%% Creating SUs
SUpositions = ceil(rand(NumberOfSUs,2)*gridsize);

%% creating original spectrum map
PUx=randperm(gridsize);PUx=PUx(1:NumberOfPUs);
PUy=randperm(gridsize);PUy=PUy(1:NumberOfPUs);
PUpower = Pmin +  rand(1,NumberOfPUs) * (Pmax-Pmin) ;   % in Watt
PUpower = 10.^(PUpower/10);
[X,Y] = meshgrid(1:resol:gridsize);
ActivePUs = ones(1,NumberOfPUs);


changeRecord = zeros(1,simTime);


%% traning phase
h = ones([size(X) NumberOfPUs]);
for t = 1: Window  
    %new map

    % Shadowing
%             a = 0.9995;
%             w = complex(randn([size(X) NumberOfPUs]) , randn([size(X) NumberOfPUs]))*shadowing;
%             if(shadowing)
%                 h = a*h +sqrt(1-a^2)*w;
%             end
%             shadowingmap = abs(h).^2
    shadowingmap = randn([size(X) NumberOfPUs])*shadowing;
    shadowingmap = 10.^(shadowingmap/10);

    IM = zeros(size(X));
    IMshadow = zeros(size(X));
    %% Markov model
    for j=1:NumberOfPUs 
        if (ActivePUs(j) == 1 )             %PU is active
            if (rand < 1-Persistence(j))
                ActivePUs(j) = 0;            %active to silent
                changeRecord(t) = 1;
            end
        else                                %PU is inactive
            if (rand < Activation(j))
                ActivePUs(j) = 1;
                changeRecord(t) = 1;
                PUpower(j) = Pmin +  rand * (Pmax-Pmin) ;   % in Watt
                PUpower(j) = 10.^(PUpower(j)/10);
            end
        end

        if (ActivePUs(j) == 0)              %PU is inactive
            continue;
        end
        
        x=PUx(j);
        y=PUy(j);
        distance = max(1, sqrt((X-x).^2+(Y-y).^2));
        IM = IM + PUpower(j)./((distance).^alpha);
        IMshadow = IMshadow + shadowingmap(:,:,j).*PUpower(j)./((distance).^alpha);
    end
    IMshadow = IMshadow*PL0;
    IM = IM*PL0;
    states(:,t) = ActivePUs.*PUpower;

    %% new measurements
    ind = zeros(1,NumberOfSUs);
    for i = 1:NumberOfSUs
        SUmap = zeros(size(IM));
       SUmap((X(1,:)==SUpositions(i,1)),(Y(:,1)==SUpositions(i,2))) = 1 ;
       ind(i) = find(SUmap);
    end
    NewMeas = IMshadow(ind);
    Pinput(:,t) = IM(ind);             
    Pshadow(:,t) = NewMeas;

    ActiveSUs = randperm(NumberOfSUs);  
    ActiveSUs = ActiveSUs(1:NumberOfActiveSUs);

    KnownData(ActiveSUs,t) = 1;
end

%% add noise , clip it and omit unobserved data
%         Pnoisy = Pshadow.*10.^(randn(size(Pinput))*0/10);
if shadowing == 0
    Pnoisy = Pinput + randn(size(Pinput))*sqrt(noise);
else
     Pnoisy = Pshadow + randn(size(Pinput))*sqrt(noise);
end

x = Pnoisy;
Pnoisy (KnownData==0) = nan;
Pnoisy = Pnoisy(:,1:Window);
Pnoisy(Pnoisy<=0) = 1e-6;

Pnoisy (KnownData(1:Window)==0) = nan;

%% Normalizing data and complete it 
mag = max(Pnoisy(:));

%% initializing alg parameters
epsl = 0.01;
maxIter = 100000;


figure(7)
subplot(3,2,1)
plot(states(1,:))
title('Original Activation #1')
subplot(3,2,2)
plot(states(2,:))
title('Original Activation #2')
%% Regular NMF
V = Pnoisy/mag; 
obs = double(~isnan(V));
[W , H] = nmfEUC(V,obs,NumberOfPUs,0,maxIter,1e-7,epsl);


figure(7)
subplot(3,2,3)
plot(H(1,:))
title('Estimated using NMF')
subplot(3,2,4)
plot(H(2,:))
title('Estimated using NMF')

%% minimiziation over beta
epsl = 1e-3;
beta = linspace(1e-5,0.5e-2,30);
maxIter = 10000;
error_dist = zeros(1,length(beta));
error_structure = zeros(1,length(beta));
for i = 1:length(beta)
    [W, H] = nmfEUC(V,obs,NumberOfPUs,beta(i),maxIter,1e-5,epsl);
    V_est = W*H;
    error_dist(i) = sum((V(KnownData==1)- V_est(KnownData==1)).^2);
    H = 1*diag(1./max(H,[],2))*H;
    structure = diff(H,[],2).^2./(diff(H,[],2).^2 + epsl);
    error_structure(i) = sum(structure(:));
end

lambda = prctile((rms(error_structure)),25);
Error_st = (error_structure).^3;
error = error_dist + beta.*Error_st;

error (rms(error_structure) >= lambda) = inf;
[~,betamin] = min(error);

%% Piecewise Constant NMF
maxIter = 100000;
beta = 1e-3;
[W , ~] = nmfEUC(V.*obs,obs,NumberOfPUs,beta,maxIter,1e-7,epsl);
H = nmfEUC_oper(V.*obs,W,obs,NumberOfPUs,beta,maxIter,1e-7,epsl);
figure(7)
subplot(3,2,5)
plot(H(1,:))
title('Estimated using PC-NMF')
subplot(3,2,6)
plot(H(2,:))
title('Estimated using PC-NMF')

saveas(gcf, '../results/plot.png');
