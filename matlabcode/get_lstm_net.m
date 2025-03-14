function layers=get_lstm_net(wd)

%网络架构
numFeatures=wd;
numResponses=1;
numHiddenUnits=300;

layers=[sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    dropoutLayer(0.1)
    lstmLayer(2*numHiddenUnits)
    dropoutLayer(0.1)
    fullyConnectedLayer(numResponses)
    regressionLayer];

end
