clear all
close all
fclose all
clc

%% Train using existed .mat

load('traderDDPGAgent.mat', 'agent');
load('trader_env.mat', 'env');
load('trainStats.mat', 'trainStats');

evl = rlEvaluator( ...
    NumEpisodes = 3, ...
    EvaluationFrequency = 50);

trainOpts = rlTrainingOptions(...
    MaxEpisodes = 30000,...
    MaxStepsPerEpisode = 20, ...
    StopTrainingCriteria = "AverageReward", ...
    StopTrainingValue = 2000, ...
    UseParallel = false );

trainStats = train(agent, env, trainStats);

save trader_env.mat env
generatePolicyFunction(agent);
% experience = sim(env,agent);

% get the agent's actor, which predicts next action given the current observation
actor_got       = getActor(agent);
% get the actor's parameters (neural network weights)
actorParams = getLearnableParameters(actor_got);