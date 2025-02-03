from StackGP import evolve, printGPModel, sortModels, allOps
import numpy as np
inputData=np.array([np.random.rand(100),np.random.rand(100),np.random.rand(100)])
responseData=inputData[0]*inputData[1]*np.sin(inputData[2])
print("Evolution initiated...")
models=evolve(inputData,responseData,ops=allOps())
print("Evolution Completed")

print("Best evolved model: ",printGPModel(sortModels(models)[0]))
print("Best evolved model error: ",sortModels(models)[0][-1])
