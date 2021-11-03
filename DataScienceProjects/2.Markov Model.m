%MARKOV MODEL EXAMPLE:

%In probability theory, a Markov model is a stochastic model used to model pseudo-randomly changing systems.
%It is assumed that future states depend only on the current state, not on the events that occurred before 
%it (that is, it assumes the Markov property). Generally, this assumption enables reasoning and computation 
%with the model that would otherwise be intractable. For this reason, in the fields of predictive modelling and probabilistic 
%forecasting, it is desirable for a given model to exhibit the Markov property.

%Matrici delle probabilitÃ 

P_74_nd=[ .9751    .0062    0       .0188;...
          .5       0        .3      .2;...
          0        0        .9812   .0188;...
          0        0        0       1];
 
P_74_cd=[ .9778    .0034    0       .0188;...
          .5       0        .3      .2;...
          0        0        .9812   .0188;...
          0        0        0       1];

P_84_nd=[ .9332    .0115    0       .0553;...
          .5       0        .3      .2;...
          0        0        .9447   .0553;...
          0        0        0       1];
 
P_84_cd=[ .9383    .0064    0       .0553;...
          .5       0        .3      .2;...
          0        0        .9447   .0553;...
          0        0        0       1];
      
      
P_90_nd=[ .8163    .0162    0       .1675;...
          .5       0        .3      .2;...
          0        0        .8325   .1675;...
          0        0        0       1];
 
P_90_cd=[ .8236    .0089    0       .1675;...
          .5       0        .3      .2;...
          0        0        .8325   .1675;...
          0        0        0       1];
      
%Matrici dei costi
C_nd=[ 0      11500    0       0;...
       0      0        6000    570;...
       0      0        6000    570;...
       0      0        0       0];
 
C_cd=[ 500    12000    0       0;...
       500    0        6500    570;...
       0      0        6500    570;...
       0      0        0       0];
   
%Matrice dei LY
%LY=[1    .3   .3   0;...
  % 1    .3   .3   0;...
 %  1    .3   .3   0;...
 %  0    0   0   0];

LY=[1    1   1   0;...
   1    1   1   0;...
   1    1   1   0;...
   0    0   0   0];

%Popolazione
v0=[ 100    0    0   0];


%%%%Soluzione

% Parametri iniziali
Tot_C_nd=0;
Tot_C_cd=500*1000;
Tot_LY_nd=0;
Tot_LY_cd=0;
v_nd=v0;
v_cd=v0;

% Ciclo for per ogni anno
N=90-65;
dC = zeros(N,1);
dLY=zeros(N,1);
dICER=zeros(N,1);

%1 fascia 66-74
for i=1:9
%Matrici dei cambiamenti di stato
T_74_nd=diag(v_nd)*P_74_nd;
T_74_cd=diag(v_cd)*P_74_cd;
%Costi di transizione
Ct_nd=T_74_nd.*C_nd;
Ct_cd=T_74_cd.*C_cd;

%Totale costi
Tot_C_nd=sum(Ct_nd, 'all')+Tot_C_nd;
Tot_C_cd=sum(Ct_cd, 'all')+Tot_C_cd;

%Calcolo LY
LY_nd=T_74_nd.*LY;
LY_cd=T_74_cd.*LY;
Tot_LY_nd=sum(LY_nd,'all')+Tot_LY_nd;
Tot_LY_cd=sum(LY_cd, 'all')+Tot_LY_cd;

v_nd=v_nd*P_74_nd;
v_cd=v_cd*P_74_cd;

dC(i) = Tot_C_cd-Tot_C_nd;
dLY(i)=Tot_LY_cd-Tot_LY_nd;
dICER(i)=(Tot_C_cd-Tot_C_nd)/(Tot_LY_cd-Tot_LY_nd);
end

%2 fascia 75-84
for i=10:20
%Matrici dei cambiamenti di stato
T_84_nd=diag(v_nd)*P_84_nd;
T_84_cd=diag(v_cd)*P_84_cd;
%Costi di transizione
Ct_nd=T_84_nd.*C_nd;
Ct_cd=T_84_cd.*C_cd;

%Totale costi
Tot_C_nd=sum(Ct_nd, 'all')+Tot_C_nd;
Tot_C_cd=sum(Ct_cd, 'all')+Tot_C_cd;

%Calcolo LY
LY_nd=T_84_nd.*LY;
LY_cd=T_84_cd.*LY;
Tot_LY_nd=sum(LY_nd,'all')+Tot_LY_nd;
Tot_LY_cd=sum(LY_cd, 'all')+Tot_LY_cd;

v_nd=v_nd*P_84_nd;
v_cd=v_cd*P_84_cd;

dC(i) = Tot_C_cd-Tot_C_nd;
dLY(i)=Tot_LY_cd-Tot_LY_nd;
dICER(i)=(Tot_C_cd-Tot_C_nd)/(Tot_LY_cd-Tot_LY_nd);
end

%3 fascia 85-89
for i=21:25
%Matrici dei cambiamenti di stato
T_90_nd=diag(v_nd)*P_90_nd;
T_90_cd=diag(v_cd)*P_90_cd;
%Costi di transizione
Ct_nd=T_90_nd.*C_nd;
Ct_cd=T_90_cd.*C_cd;

%Totale costi
Tot_C_nd=sum(Ct_nd, 'all')+Tot_C_nd;
Tot_C_cd=sum(Ct_cd, 'all')+Tot_C_cd;

%Calcolo LY
LY_nd=T_90_nd.*LY;
LY_cd=T_90_cd.*LY;
Tot_LY_nd=sum(LY_nd,'all')+Tot_LY_nd;
Tot_LY_cd=sum(LY_cd, 'all')+Tot_LY_cd;

v_nd=v_nd*P_90_nd;
v_cd=v_cd*P_90_cd;

dC(i) = Tot_C_cd-Tot_C_nd;
dLY(i)=Tot_LY_cd-Tot_LY_nd;
dICER(i)=(Tot_C_cd-Tot_C_nd)/(Tot_LY_cd-Tot_LY_nd);
end

Delta_costi=Tot_C_cd-Tot_C_nd
Delta_LY=Tot_LY_cd-Tot_LY_nd
ICER=(Tot_C_cd-Tot_C_nd)/(Tot_LY_cd-Tot_LY_nd)

x=linspace(0,25);
y=50000*x;
plot(dLY, dC,'o', x, y)
title('Relationship Delta LY and Delta Cost from 65-90 years')
xlabel('Delta LY')
ylabel('Delta Cost')
