function action1 = evaluatePolicy(observation1)
%#codegen

% Reinforcement Learning Toolbox
% Generated on: 12-Dec-2023 06:19:15

persistent policy;
if isempty(policy)
	policy = coder.loadRLPolicy("agentData1.mat");
end
% evaluate the policy
action1 = getAction(policy,observation1);