%% Generate Water-filling samples
clc
clear all
%% Number of channels
N_c=10;              
%% Other parameters
N_tr=100;        % Number of total samples, should be larger than the sum of the number of training and validation samples   
N_te=1000;         % Number of test samples
Psum=10;           % Maximal power of BS

%% Initial matrix
H_tr=zeros(N_c,N_tr); H_te=zeros(N_c,N_te);
P_tr=zeros(N_c,N_tr); P_te=zeros(N_c,N_te);
R_tr=zeros(1,N_tr);   R_te=zeros(1,N_te);

%% Generate Channel (complex Gaussian distribution)
for i_s=1:N_tr
    for j=1:N_c
      H=(randn+randn*i)/sqrt(2);
      H_tr(j,i_s)=norm(H)^2;
    end
end
for i_s=1:N_te
    for j=1:N_c
      H=(randn+randn*i)/sqrt(2);
      H_te(j,i_s)=norm(H)^2;
    end
end

%% Water-filling
for i_s=1:N_tr
   P_tr(:,i_s)=Water_filling(H_tr(:,i_s),N_c,Psum);     
end
for i_s=1:N_te
   P_te(:,i_s)=Water_filling(H_te(:,i_s),N_c,Psum);     
end
%% Data rate
for i_s=1:N_te
   R_te(:,i_s)=sum(log2(1+P_te(:,i_s).*H_te(:,i_s))); 
end
for i_s=1:N_tr
   R_tr(:,i_s)=sum(log2(1+P_tr(:,i_s).*H_tr(:,i_s))); 
end
save(['../Data/','Trainingdata_Nc',num2str(N_c)])

%% Bi-section method for finding lamda
function P  = Water_filling(H,N_c,Psum)
low=10^-6;
high=10^6;
error=10^-5; 
sum_P=0;
for i_bina=1:100
   medium=(low+high)/2;
   P=1/medium-1./H;
   P(P<0)=0;
   sum_P=sum(P);
    if abs(Psum-sum_P)<=error;
        lamda=medium;
        break;
    else if Psum-sum_P<=0
           low=medium;
        else  high=medium;
        end
    end
    end
  lamda=medium;  
end



