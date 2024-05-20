import numpy as np

def model(length,n):    # implement the actual lenght on this part
    summa=length[n]/sum(length)
    return summa

def logit(beta,time):   #the implemented dummy logit model for route choice, make it more generate, calculate in graph levelbookd
    utility=list(map(lambda x: np.exp(x*beta) ,time))
    summa = [model(utility, j) for j in range(len(time))]
    i=np.random.choice(list(range(len(time))),p=summa)    
    return i