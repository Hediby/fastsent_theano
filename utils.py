import numpy as np

def loadModel(file,max_voc="inf",decale=0):
    model = open(file)
    #floats = np.zeros((0, 200), dtype=float)
    floatsList=[]
    words = {}
    count = 0

    for line in model:    
        if len(line)< 1:
            continue
        if max_voc!="inf"and count>=max_voc:
            break
        u = line[:-1].decode('utf-8').split(" ")
        if u[0]=="":
            continue
        words[u[0]] = count
        #floats[count] = map(float, u[1:])
        #floats=np.vstack([floats,map(float, u[1:])])
        floatsList.append(map(float, u[(decale+1):]))
        if count%100000==0:
            print "processed: "+str(count)
        count+=1
                
    floats=np.asarray(floatsList)
    #floats =  floats / np.linalg.norm(floats, axis=1)[:,np.newaxis]
    return words,floats