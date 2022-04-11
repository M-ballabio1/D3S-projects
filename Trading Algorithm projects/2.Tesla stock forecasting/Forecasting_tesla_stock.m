%% IMPORTO DATASET

T = readcsv('TSLA_stock.xlsx');

%% Analisi esplorativa
% PREZZO AZIONI
% media mobile a 15gg e 25gg

ema15 = movavg(T(:,'Close'),'exponential',10);
ema25 = movavg(T(:,'Close'),'exponential',25);

f1 = figure('Position',[100,100,1250,675])  %Scelta dimensioni
plot(T.Date,T.Close,"LineWidth",1.3)
xlabel('Tempo [giorni]')
ylabel('Prezzo per azione [mln di tonnellate]')
title('Grafico azioni Tesla. Anni: 2019-2022')
hold on
plot(T.Date,ema15.Close,'r');
plot(T.Date,ema25.Close,'g');
legend('Price','10-Day Media mobile','25-Day Media mobile')
grid minor
saveas(f1,[pwd '\immagini\01.PrezzoAzioni.png'])

% VOLUMI SCAMBIATI

ema15_vol = movavg(T(:,'Volume'),'exponential',10);
ema25_vol = movavg(T(:,'Volume'),'exponential',15);

f2 = figure('Position',[100,100,1250,675])  %Scelta dimensioni
plot(T.Date,T.Volume,"LineWidth",1.3)
xlabel('Tempo [giorni]')
ylabel('Prezzo per azione [mln di tonnellate]')
title('Grafico azioni Tesla. Anni: 2019-2022')
hold on
plot(T.Date,ema15_vol.Volume,'r');
plot(T.Date,ema25_vol.Volume,'g');
legend('Price','10-Day Media mobile','15-Day Media mobile')
grid minor
saveas(f2,[pwd '\immagini\02.Volumi.png'])

%% Forecasting 

%adj fuller sulla serie vera
[h,p,adfstat,critval] = adftest(T.Close,'model','TS','lags',0:6)  %Non stazionaria (H0)

%differencing time series
N = length(T.Close);
D1 = LagOp({1,-1},'Lags',[0,1]);
dY = filter(D1,T.Close);

f3 = figure('Position',[100,100,1250,675])  %Scelta dimensioni
plot(2:N,dY)
xlim([0,N])
xlabel('Tempo [giorni]')
ylabel('Prezzo per azioni(1) - Prezzo per azioni(0) [€] ')
title('First Differenced Tesla Series')
grid minor
legend('Price')
saveas(f3,[pwd '\immagini\03.Differenze prime - TSLA.png'])

%adj fuller serie differenziata
[h,p,adfstat,critval] = adftest(dY,'model','TS','lags',0:48)  %Diventa fortemente stazionario (H1)

[h,pValue,stat,cValue] = archtest(dY)  


% AUTOREGRESSIVI: Caratteristiche grafiche della serie: autocorrelazioni e distribuzione
f4 = figure('Position',[100,100,1250,675])
% Serie storica
subplot(2,2,1)      
plot(2:N,dY);
title('Serie storica delle emissioni di C0_2')
% Istogramma della distribuzione
subplot(2,2,2)       
histfit(dY,20,'Normal')
title('Istogramma della distribuzione')
% Autocorrelazioni
subplot(2,2,3)       
autocorr(dY, 96);
title('ACF delle innovazioni')
% Autocorrelazioni parziali
subplot(2,2,4)       
parcorr(dY, 96);
title('PACF delle innovazioni')
saveas(f4,[pwd '\immagini\4.ACF_PACF_TSLA.png'])

f5 = figure('Position',[100,100,1250,675])
% Serie storica
% Autocorrelazioni
subplot(2,1,1)       
autocorr(dY, 360);
title('ACF delle innovazioni')
% Autocorrelazioni parziali
subplot(2,1,2)       
parcorr(dY, 360);
title('PACF delle innovazioni')
saveas(f5,[pwd '\immagini\5.ACF_PACF_TSLA_augmented.png'])


%% Modelli per ETEROSCHEDASTICITA'
%% modello garch(1,2)
m0 = garch(1,2)
[mhat,covM,logL] = estimate(m0,dY)
condVhat = infer(mhat,dY);             %estraggo residui condizionati ossia e^(2).
condVol = sqrt(condVhat);               %conditional volatility
% AIC e BIC
[a,b] = aicbic(logL,mhat.P+mhat.Q,N)         %aic=5.647*10^3

% Plot dei valori fittati
f6 = figure('Position',[100,100,1250,675])
plot(2:N,dY)
hold on;
plot(2:N,condVol)
title('Conditional volatility con GARCH(1,2)')
xlabel('Time')
legend('Diff Share Price TSLA','Estim. cond. volatility')
hold off;
saveas(f6,[pwd '\immagini\6.GARCH(1,2).png'])

%%% Standardized residuals
std_res = dY ./ condVol;
std_res2 = std_res .^ 2;

%%% Diagnostiche sui residui standardizzati
f7 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(std_res)
title('Standardized Residuals')
subplot(2,2,2)
histogram(std_res,10)
subplot(2,2,3)
autocorr(std_res,360)
subplot(2,2,4)
parcorr(std_res,360)
saveas(f7,[pwd '\immagini\7.acf_pacf_after_residui_std_GARCH(1,2).png'])

%%% Diagnostiche sui residui standardizzati al quadrato
f8 = figure('Position',[100,100,1250,675])
subplot(2,2,1)
plot(std_res2)
title('Standardized Residuals Squared')
subplot(2,2,2)
histogram(std_res2,10)
subplot(2,2,3)
autocorr(std_res2,360)
subplot(2,2,4)
parcorr(std_res2,360)
saveas(f8,[pwd '\immagini\8.acf_pacf_after_residui_std_quadr_GARCH(1,2).png'])
% I residui standardizzati al quadrato presentano bassissima autocorrelazione


%% modello egarch(1,3)

m1 = egarch(1,3)
[mhat1,covM1,logL1] = estimate(m1,dY)
condVhat1 = infer(mhat1,dY);             %estraggo residui condizionati ossia e^(2).
condVol1 = sqrt(condVhat1);               %conditional volatility
% AIC e BIC
[a1,b1] = aicbic(logL1,mhat1.P+mhat1.Q,N)  %AIC = 5,601*10^3

%previsione
numPeriods = 365;
vF = forecast(mhat1,numPeriods,dY);
vf_smooth= smooth(vF)
vF2 = forecast(mhat1,numPeriods);   %Previsione della conditional volatility senza esempi prima
dates = 2:N;

f9 = figure('Position',[100,100,1250,675])
plot(2:N,condVhat1,'b:','LineWidth',2);
hold on;
plot(dates(end):dates(end) + 365,[condVhat1(end);vF],'r','LineWidth',2);
plot(dates(end):dates(end) + 365,[condVhat1(end);vF2],':','LineWidth',2);
plot(dates(end):dates(end) + 365,[condVhat1(end);vF+0.2*vf_smooth], 'r--');
plot(dates(end):dates(end) + 365,[condVhat1(end);vF-0.2*vf_smooth], 'r--');
title('Forecasted Conditional Variances of TSLA next year');
ylabel('Conditional variances');
xlabel('Year');
legend('Diff Share Price TSLA','Estim. cond. volatility with Presamples','Estim. cond. volatility without Presamples','limite superiore','limite inferiore');
saveas(f9,[pwd '\immagini\9.Previsione_EGARCH(1,3).png'])

% Plot dei valori fittati
f10 = figure('Position',[100,100,1250,675])
plot(2:N,dY)
hold on;
plot(2:N,condVol1)
title('Conditional volatility con EGARCH(1,3)')
xlabel('Time')
legend('Diff Share Price TSLA','Estim. cond. volatility')
hold off;
saveas(f10,[pwd '\immagini\10.EGARCH(1,3).png'])

%% GJR-GARCH (Glosten-Jagannathan-Runkle-Generalized Autoregressive Conditional Heteroscedastic)

m2 = gjr('GARCHLags',1,'ARCHLags',1,'LeverageLags',1)
%m2.Distribution = struct('Name','t','DoF',5)
[mhat2,covM2,logL2] = estimate(m2,dY);
condVhat2 = infer(mhat2,dY);             %estraggo residui condizionati ossia e^(2).
condVol2 = sqrt(condVhat2);               %conditional volatility
% AIC e BIC
[a2,b2] = aicbic(logL2,mhat2.P+mhat2.Q,N)    %AIC = 5,61*10^3

f11 = figure('Position',[100,100,1250,675])
plot(dates,dY)
hold on;
plot(dates,condVol2)
title('Conditional volatility con GJR-GARCH(1,1)')
xlabel('Time')
legend('Prices','Estim. cond. volatility','Location','NorthEast')
hold off;
saveas(f11,[pwd '\immagini\11.GJR-GARCH(1,1).png'])






%% Modelli autoregressivi serie storiche

pMax = 3;
qMax = 3;
AIC = zeros(pMax+1,qMax+1);
BIC = zeros(pMax+1,qMax+1);

for p = 0:pMax
    for q = 0:qMax
        % White noise: ARMA(0,0)
        if p == 0 & q == 0
            Mdl = arima(0,0,0);
        end
        % Moving average: ARMA(0,q)
        if p == 0 & q ~= 0
            Mdl = arima('MALags',1:q);
        end
        % Autoregressive: ARMA(p,0)
        if p ~= 0 & q == 0
            Mdl = arima('ARLags',1:p);
        end
        % Autoregressive moving average: ARMA(p,q)
        if p ~= 0 & q ~= 0
            Mdl = arima('ARLags',1:p,'MALags',1:q);
        end      
        % Stima del modello con MLE
        EstMdl = estimate(Mdl,dY,'Display','off');
        % Salvataggio AIC e BIC
        results = summarize(EstMdl);
        AIC(p+1,q+1) = results.AIC;         % p = rows
        BIC(p+1,q+1) = results.BIC;         % q = columns
    end
end

% Confrontiamo AIC e BIC dei valori modelli stimati
minAIC = min(min(AIC))                          % minimo per riga e poi minimo per colonna della matrice AIC
[bestP_AIC,bestQ_AIC] = find(AIC == minAIC)     % posizione del modello con minimo AIC
bestP_AIC = bestP_AIC - 1; bestQ_AIC = bestQ_AIC - 1; 
minBIC = min(min(BIC))
[bestP_BIC,bestQ_BIC] = find(BIC == minBIC)
bestP_BIC = bestP_BIC - 1; bestQ_BIC = bestQ_BIC - 1; 
fprintf('%s%d%s%d%s','The model with minimum AIC is ARIMA((', bestP_AIC,',0,',bestQ_AIC,')');
fprintf('%s%d%s%d%s','The model with minimum BIC is ARIMA((', bestP_BIC,',0,',bestQ_BIC,')');
% Scegliamo il modello più parsimonioso: ARIMA((2,0,2),(12,0,0))
% Parsimonioso inteso come minor numero di parametri (Rasoio di Occam)
% In generale BIC penalizza di più la verosimiglianza quindi è più 
% parsimonioso (modello più semplice con minor numero di parametri).
ARIMA_opt = arima('ARLags',2,'MALags',2);
Est_ARIMA_opt = estimate(ARIMA_opt,dY);
E = infer(Est_ARIMA_opt, dY, 'Y0',dY(1:5));
fittedARIMA_opt = dY + E;

fittedARIMA_smooth = movavg(fittedARIMA_opt,'exponential',3);

%%% Grafico della serie osservata e fittata con ARIMA(2,0,2)
figure
plot(dates,dY)
hold on
plot(dates,fittedARIMA_opt)
legend('Osservata','ARIMA(2,0,2)','Smooth')
title('Serie storica osservata e fittata con SARIMA((1,0,0),(12,0,0))')

%valutazioni
RMSE0 = sqrt(mean((dY - fittedARIMA_opt).^2))  % Root Mean Squared Error = 20.838
RMSE1 = sqrt(mean((dY - fittedARIMA_smooth).^2))  % Root Mean Squared Error = 11.911 

