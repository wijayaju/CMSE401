import dill
import os
import sys
import time

def checkpointSave(name,data):
    file=open(str(name),"wb+")
    dill.dump(data,file)
    file.close()

def checkpointLoad(name,data):
    if os.path.exists(str(name)):
        print("\n Checkpoint Loading... \n")
        with open(str(name),'rb') as file:
            data=dill.load(file)
            print("\n Loaded Data: ",data,"\n")
    else:
        return data
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name=sys.argv[1]
    else:
        name=0
    data = 0
    data=checkpointLoad(name, data)
    while data<100:
        data+=1
        if data%1==0:
            checkpointSave(name, data)
        print("Data=",data)
        time.sleep(1)

