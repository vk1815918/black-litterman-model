from collections import OrderedDict
import pandas as pd
import numpy as np
from BLP import Model
      
        
def writeResults(filename, models):
    writeDf = pd.concat([models[0].framer.prior] + [m.df for m in models])
    writeDf.to_csv(filename, header=False)
    
assetInfo = {'Communication Services':.1,'Consumer Discretionary':.09,'Consumer Staples':.09,'Energy':.09,'Financials':.09,'Health Care':.09,'Industrials':.09,'Information Technology':.09,'Materials':.09,'Real Estate':.09,'Utilities':.09}
assetClasses = list(assetInfo.keys())
assetWeights = list(assetInfo.values())

data = pd.read_csv(r'C:\Users\vkotr\black litterman model\10yBackData.csv', usecols=assetClasses)
covMatrix = data.cov()

# 

print('Computing models...')

P_value=np.asarray([[1,0,1,0,1,0,1,0,1,0,1]])
Q_value=np.asarray([[0.015]])


# for i in range(1): 
    #len(data.columns)
model = Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=200, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.1, 
    P=P_value,
    Q=Q_value,
    identifier=1
)

# print(model.optimalPortfolio['weights'])
optimalWeights = model.optimalPortfolio['weights']
assetWeightsDict = {asset: weight for asset, weight in zip(assetClasses, optimalWeights)}
sharpe = model.optimalPortfolio['sharpe']

for asset, weight in assetWeightsDict.items():
    print(f'{ asset }: { weight }')

print(f'portfolio Sharpe ratio: { sharpe }')



model_two = Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.01, 
    P=np.asarray([[1,0,1,0,1,0,1,0,1,0,1]]),
    Q=np.asarray([[0.015]]),
    identifier=2
)

model_three = Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.01, 
    P=np.asarray([[0,0,0,0,0,0,0,0,1,0,0]]),
    Q=np.asarray([[0.015]]),
    identifier=3
)
# print(model_one)
models = (model_two, model_three)
outFile = 'new_example_output.csv'
writeResults(outFile, models)

print('Done.')
print(f'Check the model results in { outFile }')
