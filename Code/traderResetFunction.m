function [InitialObservation, InitialState] = traderResetFunction()
% Reset function to place custom environment into a random initial state.
original_cash = 10000; % range [ 0 to 10000]
position_max_cnt = 20; % range [ 0 to 20 ]
% stock_price_range = 450-200; % range [200 to 450]

global historical_data
% historical_data_in = [historical_data, 'datetime"now"'];
historical_data_in = historical_data;

global historical_indicators
historical_indicators_in = historical_indicators;


% historical_data = load('historical_VOO_data_from2017_Close_Volume_Open_High_Low_DSG2_DSG10.mat').environment_price;
[historical_data_len, ~] = size(historical_data_in);
close_price = floor(historical_data_in(:,1));
% volume = floor(historical_data(:,2));
% open_price = floor(historical_data(:,3));
% highest_price = floor(historical_data(:,4));
% lowest_price = floor(historical_data(:,5));
DSG_2 = historical_data_in(:,6);
DSG_10 = historical_data_in(:,7);

day_end_for_training = 728; % 12/31/2020

% Initialize states: cash, position, price

current_trading_day = randi([day_end_for_training historical_data_len-1],1);

for idx = height(historical_indicators_in(:,5)) : -1 : 1
    if(current_trading_day > historical_indicators_in(idx,5) )
        indicator_row = idx;
        break
    end
end
% save the trading day for traderSteoFunction.m can readback to update 
% the next trading price
save("trading_day.mat","current_trading_day"); 
position = randi([0 position_max_cnt],1);
% cost_price = close_price(current_trading_day+1,1);

% original_asset = original_cash + position * cost_price;
% save 'original_asset.mat' original_asset

price = close_price(current_trading_day,1);
cash = original_cash - (position * price);


DSG_2_init = DSG_2(current_trading_day,1);
DSG_10_init = DSG_10(current_trading_day,1);

CPI_YOY_init = historical_indicators_in(indicator_row,1);
UNRATE_init = historical_indicators_in(indicator_row,2);
CIVPART_init = historical_indicators_in(indicator_row,3);
FEDFUNDS_init = historical_indicators_in(indicator_row,4);

% Return initial environment state variables as logged signals.
InitialState = [cash; position; price; DSG_2_init; DSG_10_init; CPI_YOY_init; UNRATE_init; CIVPART_init; FEDFUNDS_init];
InitialObservation = InitialState;

end

