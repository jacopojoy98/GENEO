import os
import matplotlib.pyplot as plt 

def accvscmp(name):
    os.chdir(name)
    with open("reportfile.txt","r") as file:
        cmp = int(file.readline().split("=")[-1])
        accs = []
        for lin in file.readlines()[1:]:
            if lin[:5] == "epoch":
                flo = float(lin.split(" ")[-2])
                accs.append(flo)
        acc = max(accs)    
    
    return cmp, acc



if __name__ == "__main__":
    startdir = "/home/jcolombini/GENEO/RUNSTEN"
    cmps = []
    accs = []
    fig, ax = plt.subplots(1)
    ax.set_xscale('log')
    for name in os.listdir(startdir):
        if name[:13] == "BENCHMARK_MLP":
            cmp,acc = accvscmp(os.path.join(startdir,name))
            cmps.append(cmp)
            accs.append(acc)
    ax.scatter(cmps,accs, color ="blue", s = 40, marker="+")
    cmps = []
    accs = []
    for name in os.listdir(startdir):
        # if name == "2025-02-26-11-22LR_0.001-SzPtt_9-NImg_750-PPImg_1":
        #     continue
        if name[:4] == "2025":
            cmp,acc = accvscmp(os.path.join(startdir,name))
            cmps.append(cmp)
            accs.append(acc)
    ax.scatter(cmps,accs, color ="red", s= 40, marker ="+")
    ax.tick_params(axis="both",direction= "inout", labelsize = 10, grid_alpha=0)
    ax.set_xlabel("complexity", size=20,  weight='bold')
    ax.set_ylabel("accuracy", size= 20,  weight='bold')
    ax.set_title("Complexity vs Accuracy curve", size=20,  weight='bold')
    ax.legend(["MLP","GENEO2"])
    fig.tight_layout()
    plt.savefig("/home/jcolombini/GENEO/DEBUG_IMG/acccurveMLP.png")