%% Generate samples
clc
clear all
%% Number of files and cache size
Nf=10;  Nc=0.1*Nf;

%% Generate Zipf popularity
Para=Set_para(Nf,Nc); 
N_req=20*Para.Nf;    % Number of all requests for one realization
N_tr=20000;          % Number of total samples, should be larger than the sum of the number of training and validation samples
N_te=1000;           % Number of test set
X_ini=zeros(Para.Nf,N_tr+N_te);     % Popularity
pf_train=zeros(Para.Nf,N_tr);       % Popularity for total training set
pf_test=zeros(Para.Nf,N_te);        % Popularity for test set

%% Generate requests according to Zipf distribution
for i=1:N_tr+N_te
   Rf=randperm(Para.Nf);  Para.pf=Para.pf(Rf);
   for j=1:N_req
       s=0; flag=rand;
       for f=1:Para.Nf
          s=s+Para.pf(f);
          if flag<=s
             X_ini(f,i)=X_ini(f,i)+1;break
          end
       end     
   end    
   X_ini(:,i)=X_ini(:,i)./sum(X_ini(:,i));   
end
pf_train(:,:)=X_ini(:,1:N_tr); pf_test=X_ini(:,N_tr+1:N_tr+N_te);
X_train=X_ini(:,1:N_tr);  X_test=X_ini(:,N_tr+1:N_tr+N_te);
 
%% Generate samples according to water-filling algorithm
% Initialization
pol_tr=zeros(Para.Nf,N_tr);  pol_te=zeros(Para.Nf,N_te);
ps_te=zeros(1,N_te);  
% Water-filling algorithm
for i=1:N_tr
    Para.pf=pf_train(:,i);               
    sita=opt_ove_sita(Para);   
    Para.q=((Para.pf*Para.Z2*(1-Para.p0)/sita).^(1/2)-(1-Para.p0)*Para.Z2)/((Para.Z1-Para.Z2)*(1-Para.p0)+1);   
    Para.q(Para.q<0)=0; Para.q(Para.q>1)=1;               
    pol_tr(:,i)=Para.q'; 
end
for i=1:N_te
    Para.pf=pf_test(:,i);               
    sita=opt_ove_sita(Para);   
    Para.q=((Para.pf*Para.Z2*(1-Para.p0)/sita).^(1/2)-(1-Para.p0)*Para.Z2)/((Para.Z1-Para.Z2)*(1-Para.p0)+1);   
    Para.q(Para.q<0)=0; Para.q(Para.q>1)=1;
    ps_te(i)=sum(Para.pf.*Para.q./((1-Para.p0)*Para.q*Para.Z1+(1-Para.p0)*(1-Para.q)*Para.Z2+Para.q));                 
    pol_te(:,i)=Para.q'; 
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
