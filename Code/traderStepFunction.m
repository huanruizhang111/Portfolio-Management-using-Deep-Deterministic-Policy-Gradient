function [NextObs,normalized_Reward,IsDone,NextState] = traderStepFunction(Action,State)
% Custom step function to construct trader environment for the function
% name case.
% states: profit, cash, position, price

% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

% Define the environment constants.
maxReward = 10000;
minReward = -10000;
MaxTradeCnt = 20;
original_cash = 10000;
% original_asset = load('original_asset.mat').original_asset;

global historical_data
% historical_data = [historical_data, 'datetime"now"'];
historical_data_in = historical_data;

global historical_indicators
historical_indicators_in = historical_indicators;

% historical_data = load('historical_VOO_data_from2017_Close_Volume_Open_High_Low_DSG2_DSG10.mat').environment_price;
% [historical_data_len, ~] = size(historical_data);
close_price = floor(historical_data_in(:,1));
% volume = floor(historical_data(:,2));
% open_price = floor(historical_data(:,3));
% highest_price = floor(historical_data(:,4));
% lowest_price = floor(historical_data(:,5));
DSG_2 = historical_data_in(:,6);
DSG_10 = historical_data_in(:,7);

day_end_for_training = 728; % 12/31/2020
current_trading_day = load('trading_day.mat').current_trading_day;

for idx = height(historical_indicators_in(:,5)) : -1 : 1
    if(current_trading_day > historical_indicators_in(idx,5) )
        indicator_row = idx;
        break
    end
end

% Reward each time step the cart-pole is balanced
% RewardForNotFalling = 1;
% Penalty when the cart-pole fails to balance
% PenaltyForFalling = -10;

% Unpack the state vector from the logged signals.
cash = State(1);
position = State(2);
price = State(3);

% Check if the given action is valid.
% if ~ismember(Action,[-MaxTradeCnt MaxTradeCnt])
%     error('Action must between %g and %g.',...
%         -MaxTradeCnt,MaxTradeCnt);
% end
% if( (cash - (price * Action)) < 0 )
%     error('Shortage in cash with amount of %g.',...
%         cash - (price * Action));
% end

% Cache to avoid recomputation.

if( (Action >= (-MaxTradeCnt)) && ...
    (Action <= MaxTradeCnt) && ...
    (position + Action) > 0 && ...
    ( (cash - (price * Action)) >= 0 ) )

    % Calculate next state.
    next_cash = cash - (price * Action);
    next_position = position + Action;
    
    % if(current_trading_day - 1 >= day_end_for_training)
    %     next_price = close_price(current_trading_day - 1 ,1); % read from next day
    %     IsDone = 0;
    % else
    %     next_price = price;% break for trading day less than day_end_for_training
    %     IsDone = 1;
    % end
    
    % Calculate reward.
    % calculate the unrealized profit and remaining cash minus original cash
    % to be the reward of this action.
    % Reward = (next_cash + (next_position * price) ) - original_asset;
    Reward = 1 * ((next_cash + (next_position * price) ) - original_cash );
else
    next_cash = cash;
    next_position = position;

    % if(current_trading_day - 1 >= day_end_for_training)
    %     next_price = close_price(current_trading_day - 1 ,1); % read from next day
    %     IsDone = 0;
    % else
    %     next_price = price;% break for trading day less than day_end_for_training
    %     IsDone = 1;
    % end

    % Calculate reward.
    % calculate the unrealized profit and remaining cash minus original cash
    % to be the reward of this action.
    Reward = -original_cash;
end

% Normalize rewards to the range [-1, 1]
normalized_Reward = (Reward - minReward) / (maxReward - minReward) * 2 - 1;

next_price = close_price(current_trading_day - 1 ,1); % read from next day
next_DSG_2 = DSG_2(current_trading_day - 1 ,1); % read from next day
next_DSG_10 = DSG_10(current_trading_day - 1 ,1); % read from next day
if(current_trading_day - 1 > day_end_for_training)
    IsDone = 0;
else
    IsDone = 1;
end
next_CPI_YOY =  historical_indicators_in(indicator_row-1,1);
next_UNRATE =  historical_indicators_in(indicator_row-1,2);
next_CIVPART =  historical_indicators_in(indicator_row-1,3);
next_FEDFUNDS =  historical_indicators_in(indicator_row-1,4);

NextState = [next_cash; next_position; next_price; next_DSG_2; next_DSG_10; next_CPI_YOY; next_UNRATE; next_CIVPART; next_FEDFUNDS];
% Copy next state to next observation.
NextObs = NextState;

current_trading_day = current_trading_day - 1;
save("trading_day.mat","current_trading_day"); 


% Check terminal condition.
% IsDone = abs(X) > DisplacementThreshold || abs(Theta) > AngleThreshold;
% IsDone = 0;



end