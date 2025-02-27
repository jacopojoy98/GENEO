import torch
from FGENEO_batch import GENEO_thorus
import os
from funtions import plots
def explain(name):
    os.chdir(name)
    patterns = torch.load("patterns.pt")
    model = GENEO_thorus(patterns, 10)
    model.load_state_dict(torch.load("model.pt", weights_only=True ))
    
    for p in model.parameters():
        for j,k in enumerate(p):
            values, args = torch.topk(torch.abs(k), 10) 
            print(args)
            print(values)
            for i in args:
                plots(patterns[i].squeeze(),"class"+str(j)+"_pattern"+str(i))
            
            input()

if __name__ == "__main__":
    explain("/home/jcolombini/GENEO/RUNSTEN/2025-02-24-12-11LR_0.03-SzPtt_9-NImg_150-PPImg_1")