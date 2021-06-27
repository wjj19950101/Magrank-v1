%% Generate samples
clc
clear all
%% Number of files and cache size
Nf=30;  Nc=0.1*Nf;

%% Generate Zipf popularity
Para=Set_para(Nf,Nc); 
N_req=20*Para.Nf;    % Number of all requests for one realization
N_tr=340000;          % Number of total samples, should be larger than the sum of the number of training and validation samples
N_te=1000;           % Number of test set
d_past=5;
X_ini=zeros(Para.Nf,N_tr+N_te+d_past);     % Popularity
delta_t=0.6+0.2*exp(-(([1:N_tr+N_te+d_past]-1)/10000));
X_train=zeros(Para.Nf,d_past,N_tr);
X_test=zeros(Para.Nf,d_past,N_te);
pf_train=zeros(Para.Nf,N_tr);       % Popularity for total training set
pf_test=zeros(Para.Nf,N_te);        % Popularity for test set

%% Generate requests according to Zipf distribution
for t=1:N_tr+N_te+d_past   
   Para.pf=[1:Para.Nf].^-delta_t(t)./(sum([1:Para.Nf].^-delta_t(t))); 
%    Rf=randperm(Para.Nf);  Para.pf=Para.pf(Rf);
   for j=1:N_req
       s=0; flag=rand;
       for f=1:Para.Nf
          s=s+Para.pf(f);
          if flag<=s
             X_ini(f,t)=X_ini(f,t)+1;break
          end
       end     
   end    
   X_ini(:,t)=X_ini(:,t)./sum(X_ini(:,t));   
end

for i=1:N_tr
    X_train(:,:,i)=X_ini(:,i:d_past+i-1);
%     X_train(:,:,i)=X_ini(:,i+1:d_past+i);
    pf_train(:,i)=X_ini(:,d_past+i);
    % Rand Input
    Rf=randperm(Para.Nf); X_train(:,:,i)=X_train(Rf,:,i);  pf_train(:,i)= pf_train(Rf,i);
end

for i=1:N_te
    X_test(:,:,i)=X_ini(:,i+N_tr:d_past+i+N_tr-1);
%      X_test(:,:,i)=X_ini(:,i+N_tr+1:d_past+i+N_tr);
    pf_test(:,i)=X_ini(:,d_past+N_tr+i);
    % Rand Input
    Rf=randperm(Para.Nf); X_test(:,:,i)=X_test(Rf,:,i);  pf_test(:,i)= pf_test(Rf,i);
end

% pf_train(:,:)=X_ini(:,1:N_tr); pf_test=X_ini(:,N_tr+1:N_tr+N_te);
% X_train=X_ini(:,1:N_tr);  X_test=X_ini(:,N_tr+1:N_tr+N_te);
 


%% Generate samples according to water-filling algorithm
% Initialization
pol_tr=zeros(Para.Nf,N_tr);  pol_te=zeros(Para.Nf,N_te);
ps_te=zeros(1,N_te);  
% Water-filling algorithm
for t=1:N_tr
    Para.pf=pf_train(:,t);               
    sita=opt_ove_sita(Para);   
    Para.q=((Para.pf*Para.Z2*(1-Para.p0)/sita).^(1/2)-(1-Para.p0)*Para.Z2)/((Para.Z1-Para.Z2)*(1-Para.p0)+1);   
    Para.q(Para.q<0)=0; Para.q(Para.q>1)=1;               
    pol_tr(:,t)=Para.q'; 
end
for t=1:N_te
    Para.pf=pf_test(:,t);               
    sita=opt_ove_sita(Para);   
    Para.q=((Para.pf*Para.Z2*(1-Para.p0)/sita).^(1/2)-(1-Para.p0)*Para.Z2)/((Para.Z1-Para.Z2)*(1-Para.p0)+1);   
    Para.q(Para.q<0)=0; Para.q(Para.q>1)=1;
    ps_te(t)=sum(Para.pf.*Para.q./((1-Para.p0)*Para.q*Para.Z1+(1-Para.p0)*(1-Para.q)*Para.Z2+Para.q));                 
    pol_te(:,t)=Para.q'; 
end
save(['../Data/Sup_WFpol_Nf',num2str(Para.Nf)])

%% Find the optimal water-level by the method of bisection
function sita=opt_ove_sita(Para)
sita=zeros(1,1); sum_qf=0;
error=10^-5; low=10^-6; high=100;    
for i_bina=1:100
   medium=(low+high)/2;
   Para.q=((Para.pf*(Para.Z2)*(1-Para.p0)/medium).^(1/2)-Para.Z2*(1-Para.p0))/((Para.Z1-Para.Z2)*(1-Para.p0)+1);
   Para.q(Para.q<0)=0; Para.q(Para.q>1)=1;sum_qf=sum(Para.q);
    if abs(Para.Nc-sum_qf)<=error;
        sita=medium;
        break;
    else if Para.Nc-sum_qf<=0
           low=medium;
        else  high=medium;
        end
    end
end
  sita=medium;     
end

%% system parameters
function Para=Set_para(Nf,Nc)
Lamd=5/250^2/pi;                % BS density
Lamd_u=5/250^2/pi;              % User density                                  
R0=2*10^6;                      % Data rate requement
W=20*10^6;                      % Total bandwidth     
Para.Nf=Nf;                     % file number
Para.Nc=Nc;                      % Cache size
Para.q=zeros(2,Para.Nf);        % Caching policy               
Para.pf=[1:Para.Nf].^-0.6./(sum([1:Para.Nf].^-0.6)); % Zipf distribution
% Calcu constant  
gama0=2^(R0*(1+1.28*Lamd_u/Lamd)/W)-1;
Para.p0=(1+Lamd_u/3.5/Lamd)^(-3.5);
Para.Z1=gama0^(2/3.7)*quad(@(y)(1./(1+y.^(3.7/2))),gama0^(-2/3.7),1000);      
Para.Z2=gamma(1-2/3.7)*gamma(1+2/3.7)*(gamma(1))^(-1)*gama0^(2/3.7);      

end
