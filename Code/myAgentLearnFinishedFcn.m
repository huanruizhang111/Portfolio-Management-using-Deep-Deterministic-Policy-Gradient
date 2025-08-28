function dataToLog = myAgentLearnFinishedFcn(data)

    if mod(data.AgentLearnCount, 2) == 0
        dataToLog.ActorLoss  = data.ActorLoss;
        dataToLog.CriticLoss = data.CriticLoss;
    else
        dataToLog = [];
    end
    
end