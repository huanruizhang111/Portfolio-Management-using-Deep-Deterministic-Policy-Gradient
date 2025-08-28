clear all
close all
fclose all
clc

%% Define Environment
MaxTradeCnt = 20;
global historical_data
historical_data = load('historical_VOO_data_from2017_Close_Volume_Open_High_Low_DSG2_DSG10.mat').environment_price;

global historical_indicators
historical_indicators_tmp = readtable('All_indicators.xlsx');
historical_indicators = table2array(historical_indicators_tmp(:,2:6));

% The observations from the environment are the profit, cash, position, price
obsInfo = rlNumericSpec([9 1],'LowerLimit',[0; 0; 200; 0; 0; 0.001179264; 3.4; 60.1; 0.05],'UpperLimit',[20000; 20; 450; 100; 100; 0.09059758; 14.7; 63.3; 5.33]);
obsInfo.Name = "Financial Trader States";
obsInfo.Description = 'cash, position, price, DSG2, DSG10, CPI_YOY,	UNRATE,	CIVPART, FEDFUNDS';

% The environment has a discrete action space where the agent can apply 
% one of the possible trading action to the cart, –20 to 20. Create the action specification for these actions. 
% actInfo = rlFiniteSetSpec([-20 20]); % positive: # of stocks to buy. negative: # of stocks to sell. zeros: hold
% actInfo.Name = "Financial Trader Action";

actInfo = rlNumericSpec([1 1]);
actInfo.Name = "Financial Trader States";
actInfo.Description = 'Action';
actInfo.LowerLimit = -MaxTradeCnt;
actInfo.UpperLimit = MaxTradeCnt;

% actInfo = rlFiniteSetSpec([-5 -4 -3 -2 -1 0 1 2 3 4 5]);
% actInfo.Name = "Financial Trader States";
% actInfo.Description = 'Action';

% Create the custom environment using the defined observation specification, action specification, and function names.
env = rlFunctionEnv(obsInfo,actInfo,"traderStepFunction","traderResetFunction");
%% Define Critic

% % % Create an environment with a continuous action space and obtain its 
% % % observation and action specifications. For this example, load the 
% % % environment used in the example Train DDPG Agent to Control Double 
% % % Integrator System. The observation from the environment is a vector 
% % % containing the position and velocity of a mass. The action is a scalar
% % % representing a force ranging continuously from -2 to 2 Newton.
% env = rlPredefinedEnv("DoubleIntegrator-Continuous");

% Obtain the environment observation and action specification objects.
% obsInfo = getObservationInfo(env);
% actInfo = getActionInfo(env);

% The actor and critic networks are initialized randomly. Ensure reproducibility
% by fixing the seed of the random generator.
rng(0)

% Define observation and action paths
obsPath = featureInputLayer(prod(obsInfo.Dimension),Name="obsInLyr");
actPath = featureInputLayer(prod(actInfo.Dimension),Name="actInLyr");

% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(300)
    tanhLayer
    fullyConnectedLayer(200)
    tanhLayer
    % fullyConnectedLayer(8)
    % tanhLayer
    % fullyConnectedLayer(30)
    % tanhLayer
    fullyConnectedLayer(1)
    ];

% Add paths to layerGraph network
criticNet = layerGraph(obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);

% Connect paths
criticNet = connectLayers(criticNet,"obsInLyr","concat/in1");
criticNet = connectLayers(criticNet,"actInLyr","concat/in2");

% Plot the network
plot(criticNet)

% Convert to dlnetwork object
criticNet = dlnetwork(criticNet);

% Display the number of weights
summary(criticNet)

% Create the critic approximator object using criticNet, the environment observation and action specifications, and the names of the network input layers to be connected with the environment observation and action channels. For more information, see rlQValueFunction.
critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr", ...
    UseDevice = "cpu");

% Check the critic with random observation and action inputs.
getValue(critic,{rand(obsInfo.Dimension)},{rand(actInfo.Dimension)})

save traderDDPGCritic.mat critic
%% Define Actor

% DDPG agents use a parametrized deterministic policy over continuous action spaces, 
% which is learned by a continuous deterministic actor. This actor takes the current 
% observation as input and returns as output an action that is a deterministic function 
% of the observation.To model the parametrized policy within the actor, use a neural 
% network with one input layer (which receives the content of the environment observation
% channel, as specified by obsInfo) and one output layer (which returns the action to the
% environment action channel, as specified by actInfo).

% Define the network as an array of layer objects.
% Create a network to be used as underlying actor approximator
actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(48)
    tanhLayer
    fullyConnectedLayer(48)
    tanhLayer
    % fullyConnectedLayer(8)
    % tanhLayer
    % fullyConnectedLayer(30)
    % tanhLayer
    fullyConnectedLayer(prod(actInfo.Dimension))
    ];

% Convert to dlnetwork object
actorNet = dlnetwork(actorNet);

% Display the number of weights
summary(actorNet)

% Create the actor using actorNet and the observation and action specifications. For more information on continuous deterministic actors, see rlContinuousDeterministicActor.
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo, UseDevice="cpu");

% Check the actor with a random observation input.
getAction(actor,{rand(obsInfo.Dimension)})


save traderDDPGActor.mat actor
%% Define Agent

% Create the DDPG agent using the actor and critic.
agent = rlDDPGAgent(actor,critic);
agent.AgentOptions.SampleTime=0.01; % env.Ts;
agent.AgentOptions.TargetSmoothFactor=1e-2;
agent.AgentOptions.ExperienceBufferLength=1e5;
agent.AgentOptions.DiscountFactor=0.99;
agent.AgentOptions.MiniBatchSize=32;

agent.AgentOptions.CriticOptimizerOptions.LearnRate=5e-4;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold=1;

agent.AgentOptions.ActorOptimizerOptions.LearnRate=1e-3;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold=1;

save traderDDPGAgent.mat agent

% Create a policy object from actor.
% policy = rlDeterministicActorPolicy(actor);

%% Train using existed .mat

% Define an options object to train agent2 from data for 50 epochs.
% tfdOpts = rlTrainingFromDataOptions("MaxEpochs",5);

% To train agent from data, use trainFromData. Pass the fileDataStore object fds as second input argument.
mat_file_path = "D:\University_of_Victoria\MADS\Term_1\" + ...
                "ECE559B_DeepReinforementLearning\Assignments\Project\" + ...
                "MATLAB_Code\Mine\" + ...
                "historical_VOO_data_from2017_Close_Volume_Open_High_Low.mat";
fds = fileDatastore("historical_VOO_data_from2017_Close_Volume_Open_High_Low_DSG2_DSG10.mat","ReadFcn",@load,"FileExtensions",".mat");

evl = rlEvaluator( ...
    NumEpisodes = 1, ...
    EvaluationFrequency = 50);

trainOpts = rlTrainingOptions(...
    MaxEpisodes = 30000,...
    MaxStepsPerEpisode = 20, ...
    StopTrainingCriteria = "AverageReward", ...
    StopTrainingValue = 3000, ...
    UseParallel = false );

% trainFromData(agent,fds,tfdOpts);
% trainFromData(agent);

monitor = trainingProgressMonitor();
logger = rlDataLogger(monitor);
logger.AgentLearnFinishedFcn = @myAgentLearnFinishedFcn;

trainStats = train(agent, env, trainOpts, Evaluator = evl, Logger=logger);

save trader_env.mat env
save trainStats.mat trainStats
generatePolicyFunction(agent);
% experience = sim(env,agent);

% get the agent's actor, which predicts next action given the current observation
actor_got       = getActor(agent);
% get the actor's parameters (neural network weights)
actorParams = getLearnableParameters(actor_got);