from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist = []
class Domain:
    def __init__(self,_label,_nLength,_nCount, _entropy,_name):
           self.label = _label
           self.name=_name
           self.nLength = _nLength
           self.nCount = _nCount
           self.entropy = _entropy
    def returnData(self):
        return [self.nLength, self.nCount, self.entropy]
    
    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

def getNCount(dstr):
    nCount=0
    for i in dstr:
        if i.isdigit():
            nCount=nCount+1
    return nCount

def getEntropy(dstr):
    stats={}
    dlen=len(dstr)
    entropy=0
    for i in dstr:
        if stats.get(i)==None:
            stats[i]=1
        else:
            stats[i]+=1
    for i in stats:
        tp=-(stats[i]/dlen)*math.log(stats[i]/dlen,2)
        entropy=entropy+tp
    return entropy

def initTrainData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            nlen = len(tokens[0])
            nCount=getNCount(tokens[0])
            entropy=getEntropy(tokens[0])
            label = tokens[1]
            domainlist.append(Domain(label,nlen,nCount,entropy,tokens[0]))

def main():
    initTrainData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix,labelList)
    with open("test.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            t_nlen=len(line)
            t_nCount=getNCount(line)
            t_entropy=getEntropy(line)
            with open("result.txt",mode='a') as fres:
                fres.write(line+",")
                if clf.predict([[t_nlen,t_nCount,t_entropy]])==[0]:
                    fres.write("notdga"+'\n')
                else:
                    fres.write("dga"+'\n')
main()
