clear all
close all
fclose all
clc

MaxTradeCnt = 1;% 20;

load('traderDDPGAgent.mat', 'agent');
load('trader_env.mat', 'env');

% generatePolicyFunction(agent); % use this
% experience = sim(env,agent); % Don't use, this is for simulink

%% Load historical stock price for testing
historical_data = load('historical_VOO_data_from2017_Close_Volume_Open_High_Low_DSG2_DSG10.mat').environment_price;
% [historical_data_len, ~] = size(historical_data);


historical_indicators_tmp = readtable('All_indicators.xlsx');
historical_indicators = table2array(historical_indicators_tmp(:,2:6));

close_price = floor(historical_data(:,1));
% volume = floor(historical_data(:,2));
% open_price = floor(historical_data(:,3));
% highest_price = floor(historical_data(:,4));
% lowest_price = floor(historical_data(:,5));
DSG_2 = historical_data(:,6);
DSG_10 = historical_data(:,7);

start_evaluate_trading_day = 475; % 224; % 288; % 475; % 727;
indicator_row = 1;
for idx = height(historical_indicators(:,5)) : -1 : 1
    if(start_evaluate_trading_day > historical_indicators(idx,5) )
        indicator_row = idx;
        break
    end
end

% states: 'cash, position, price'
cash = 10000;
position = 10; %10;
price = close_price(start_evaluate_trading_day, 1);
cost_price = close_price(start_evaluate_trading_day+1, 1);
Original_asset = cash + position * cost_price;
GSD_2_current = DSG_2(start_evaluate_trading_day, 1);
GSD_10_current = DSG_10(start_evaluate_trading_day, 1);
CPI_YOY_current =  historical_indicators(indicator_row,1);
UNRATE_current =  historical_indicators(indicator_row,2);
CIVPART_current =  historical_indicators(indicator_row,3);
FEDFUNDS_current =  historical_indicators(indicator_row,4);

% Profit = Original_asset;
ObservationInfo = [cash ; position ; price ; GSD_2_current ; GSD_10_current; CPI_YOY_current; UNRATE_current; CIVPART_current; FEDFUNDS_current]; % St
ObservationInfo_all(start_evaluate_trading_day , :) = ObservationInfo'; % save St+1 for analysis and debug use
ProfitCumRatio_currentAsset_originalAsset(start_evaluate_trading_day+1, 1:3) = [0, Original_asset, Original_asset];

for day = start_evaluate_trading_day : -1 : 2
    current_asset = ObservationInfo(1,1) + ObservationInfo(2,1) * ObservationInfo(3,1);
    Action = evaluatePolicy(ObservationInfo) * MaxTradeCnt; % At
    % Action = min(Action, MaxTradeCnt);
    % if( (ObservationInfo(2,1) + Action < 0) )  % selling more than have. Change action to only hold
    %     Action = 0;
    % end
    Action_all(day , 1) = Action; % save At for analysis and debug use

    next_day = day - 1;
    next_cash = ObservationInfo(1,1)- (ObservationInfo(3,1) * Action);
    next_position = ObservationInfo(2,1) + Action;
    next_price = close_price(next_day, 1);
    next_GSD_2 = DSG_2(next_day, 1);
    next_GSD_10 = DSG_10(next_day, 1);
    for idx = height(historical_indicators(:,5)) : -1 : 1
        if(next_day > historical_indicators(idx,5) )
            indicator_row = idx;
            break
        end
    end
    next_CPI_YOY =  historical_indicators(indicator_row,1);
    next_UNRATE =  historical_indicators(indicator_row,2);
    next_CIVPART =  historical_indicators(indicator_row,3);
    next_FEDFUNDS =  historical_indicators(indicator_row,4);
    
    ProfitCumRatio_currentAsset_originalAsset(day, 1:3) = [ (current_asset - Original_asset)/Original_asset, current_asset, Original_asset];
    ObservationInfo = [next_cash ; next_position ; next_price ; next_GSD_2 ; next_GSD_10; next_CPI_YOY; next_UNRATE; next_CIVPART; next_FEDFUNDS]; % St+1
    ObservationInfo_all(next_day , :) = ObservationInfo'; % save St+1 for analysis and debug use
end


[Trough_VOO_val, Trough_VOO_idx] = min(close_price(2:start_evaluate_trading_day,1));
[Peak_VOO_val, Peak_VOO_idx] = max(close_price(2:start_evaluate_trading_day,1));
if(Trough_VOO_idx <= Trough_VOO_idx)
    MDD_VOO = (Trough_VOO_val - Peak_VOO_val) / Peak_VOO_val *100;
else
    MDD_VOO = 0;
end
MDD_VOO

[Trough_DDPG_val, Trough_DDPG_idx] = min(ProfitCumRatio_currentAsset_originalAsset(2:start_evaluate_trading_day,2));
[Peak_DDPG_val, Peak_DDPG_idx] = max(ProfitCumRatio_currentAsset_originalAsset(2:start_evaluate_trading_day,2));
if(Trough_DDPG_idx <= Trough_DDPG_idx)
    MDD_DDPG = (Trough_DDPG_val - Peak_DDPG_val) / Peak_DDPG_val *100;
else
    MDD_DDPG = 0;
end
MDD_DDPG

% Return_VOO = ( (close_price(2,1) / close_price(start_evaluate_trading_day, 1)) ^ (1/((start_evaluate_trading_day-2)/252))) - 1;
% Return_portfolio_DDPG = ( (ProfitCumRatio_currentAsset_originalAsset(2, 2) / ProfitCumRatio_currentAsset_originalAsset(start_evaluate_trading_day, 2)) ^ (1/((start_evaluate_trading_day-2)/252))) - 1;

annual_Return_free = 0; % 0.025;
for day = start_evaluate_trading_day : -1 : 3
    day_price_change_VOO(day-2,1) = (close_price(day-1,1)-close_price(day,1))/close_price(day,1);
    day_price_change_DDPG(day-2,1) = (ProfitCumRatio_currentAsset_originalAsset(day-1,2)-ProfitCumRatio_currentAsset_originalAsset(day,2))/ProfitCumRatio_currentAsset_originalAsset(day,2);
end
annual_Return_VOO = mean(day_price_change_VOO) * length(day_price_change_VOO);
annual_Return_portfolio_DDPG = mean(day_price_change_DDPG) * length(day_price_change_DDPG);
annual_std_VOO = sqrt(252)*std(day_price_change_VOO);
annual_std_DDPG = sqrt(252)*std(day_price_change_DDPG);

annual_Sharpe_ratio_VOO = (annual_Return_VOO - annual_Return_free) / annual_std_VOO
annual_Sharpe_ratio_DDPG = (annual_Return_portfolio_DDPG - annual_Return_free) / annual_std_DDPG

figure(1),
yyaxis left
plot(2:start_evaluate_trading_day, ProfitCumRatio_currentAsset_originalAsset(start_evaluate_trading_day:-1:2, 2))
hold on
plot(2:start_evaluate_trading_day, ProfitCumRatio_currentAsset_originalAsset(start_evaluate_trading_day:-1:2, 3))
hold on
yyaxis right
plot(2:start_evaluate_trading_day, close_price(start_evaluate_trading_day:-1:2, 1))
hold on

title('Evaluation')
legend('Current asset', 'Original asset', 'VOO Market Close Price')
legend('Location', 'southoutside');
xlabel('Trading day')
ylabel('Asset value')
hold off
% saveas(gcf, 'IterationMovement.bmp')
