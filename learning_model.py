def gawron(alpha,real_travel_time,previous_exp):
    
    cost=[]

    for i in range(len(real_travel_time)):
    
        cost.append((1-alpha)*previous_exp[i]+alpha*real_travel_time[i])

    return cost