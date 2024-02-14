#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


import gurobipy as gp
from gurobipy import GRB

from mahalanobis_distance_calculator import MDistance_Squared_SinglePoint

#%%

# prepare the top 10% of the data for formulating KNN constraints (see Section 4.4)
def TrainingDataForKNN(X_scaled, y_scaled, n_samples, pct_good_samples, predictive_model):
    X_scaled = np.round(X_scaled, 4)
    df_1 = pd.DataFrame(X_scaled)
    df_2 = pd.DataFrame(abs(predictive_model.predict(X_scaled) - y_scaled), columns = ['error'])
    df_3 = pd.DataFrame(y_scaled, columns = ['y'])
    df = pd.concat([df_1, df_2, df_3], axis=1, sort=False)
    df_trunc = df.nsmallest(int(np.floor(n_samples * pct_good_samples)), columns = ['y', 'error'])
    X_good = np.array(df_trunc.iloc[:,:-2])
    
    return X_good


#%%

def nn_unrestricted(true_function, min_max_scaler, X_train, nn, LB, UB, bigM = 10000000.0, OutputFlag = 1, timelimit = 200): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    BASE model
    """
    
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)    
    num_inputs = layers_size[0]
    n_variables = num_inputs


    my_sols = []
    my_ys = []
    
    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []

        #x_binarized = {}
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
         
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
                
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
         
        objective_expr = gp.LinExpr(0.0)
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, GRB.MINIMIZE)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit) 
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
                   
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn
                            
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
                
                
                    
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        print(my_sols[0][0])
        print("optimal prediction in scale of [0, 1]: {}".format(OptimalPredictionUnrestrictedModel))
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
    
#%%

def nn_unrestricted_real_data(nn, LB, UB, obj = GRB.MAXIMIZE, bigM = 10000000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    BASE model
    utilized for a real-world dataset
    """
    
    #layers_size,node_layer_mapping,Ws,bs,n_nodes,offset
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs
    
    my_sols = []
    my_ys = []
    
    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []

        #x_binarized = {}
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
         
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
                
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
         
        objective_expr = gp.LinExpr(0.0) 
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, obj)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit) 
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
                   
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn
                            
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
                
                
                    
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        return OptimalPredictionUnrestrictedModel, my_sols[0][0], model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
              

#%%

def nn_mdist(true_function, min_max_scaler, X_train, nn, LB, UB, CriteriaMdist, bigM = 10000000.0, OutputFlag = 1, timelimit = 200): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    MD constraints
    """
    
    #layers_size,node_layer_mapping,Ws,bs,n_nodes,offset
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs

    my_sols = []
    my_ys = []
    
    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []

        #x_binarized = {}
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
         
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
                
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
            
        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
         
        objective_expr = gp.LinExpr(0.0) 
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, GRB.MINIMIZE)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit) 
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
                   
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn

                            
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
                
                
                    
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap 
    else:
        return 'null', 'null',model.RunTime, 'null'

#%%

def nn_mdist_real_data(min_max_scaler, X_train, nn, LB, UB, CriteriaMdist, obj = GRB.MAXIMIZE, bigM = 10000000.0, OutputFlag = 1, timelimit = 200): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    MD constraints
    utilized for a real-world dataset
    """
    #layers_size,node_layer_mapping,Ws,bs,n_nodes,offset
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs


    my_sols = []
    my_ys = []
    
    try:
      
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []

        #x_binarized = {}
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
         
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
                
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
            
        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
         
        objective_expr = gp.LinExpr(0.0) 
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, obj)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit) 
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
                   
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn

                            
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
                
                
                    
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        return OptimalPredictionUnrestrictedModel, my_sols[0][0], model.RunTime, model.MIPGap 
    else:
        return 'null', 'null',model.RunTime, 'null'

# In[111]:

def nn_kNN(true_function, min_max_scaler, X_train, k_neighbors, D, nn, LB, UB, bigM_kNN=10000.0, bigM = 1000000000.0, OutputFlag = 1, timelimit = 200): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    KNN constraints
    """
    
    #layers_size,node_layer_mapping,Ws,bs,n_nodes,offset
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs


    my_sols = []
    my_ys = []
    
    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []
        
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        #x_binarized = {}
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")


# kNN relevant variables
        for j in range(num_inputs):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) # 1.0
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)

        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")
        
            
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
                
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
            

# kNN relevant constraints
        for j in range(num_inputs):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])              
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(num_inputs):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
    
            
        objective_expr = gp.LinExpr(0.0) 
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, GRB.MINIMIZE)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-6)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        

        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
            distances = []    
 
            # print("optimal solution: x") 
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn

                
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
             
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'

#%%

def nn_kNN_real_data(X_train, k_neighbors, D, nn, LB, UB, bigM_kNN=10000.0, bigM = 1000000000.0, obj = GRB.MAXIMIZE, OutputFlag = 0, timelimit = 200): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    KNN constraints
    utilized for a real-world dataset
    """
    X_train = np.array(X_train)
    
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs

    my_sols = []
    my_ys = []
    
    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []
        
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")


# kNN relevant variables
        for j in range(num_inputs):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) # 1.0
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)

        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")
        
            
        model.update()
        

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
               
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
            

# kNN relevant constraints
        for j in range(num_inputs):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])
                
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(num_inputs):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
            
        objective_expr = gp.LinExpr(0.0)
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, obj)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        

        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
            distances = []
            
            for i in range(n_variables):
                my_sol[0][i] = x[i].Xn
                
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
             
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    
    if nSolutions > 0:
        return model.ObjVal, my_sols[0][0], model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'



#%%

def nn_isolation_forest(nn, isolation_forest_info, true_function, X_train, LB, UB, min_max_scaler, L, objective = GRB.MINIMIZE, bigM_if=10000.0, bigM = 1000000000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    IF constraints
    """
    
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs


    my_sols = []
    my_ys = []
        
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']

    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []
              
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
            # z[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z[" +str(i) + "]", lb=0.0, ub=1.0)      
            
        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
                
        model.update()

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
                ###
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)

        
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= beta can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])
        
            
        objective_expr = gp.LinExpr(0.0)
        objective_expr += x[n_nodes-1]

            
        model.setObjective(objective_expr, objective)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        # print("Final MIP gap value: %f" % model.MIPGap)

        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
            
            for i in range(n_variables):
                my_sol[0][i] = round(x[i].Xn, 4)

                
            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
             
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    except AttributeError:
        print('Encountered an attribute error')    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'

# In[111]:

def nn_isolation_forest_real_data(nn, isolation_forest_info, X_train, LB, UB, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, bigM = 10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    IF constraints
    utilized for a real-world dataset
    """
    
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    
    num_inputs = layers_size[0]
    n_variables = num_inputs


    my_sols = []
    my_ys = []
        
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']

    try:
       
        model = gp.Model("base_ip")
        ### x is pre-relu
        x = {}
        sol_temp = []
              
        for i in range(n_nodes):
            if(i < num_inputs):
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=LB[i],ub=UB[i])
                sol_temp.append(x[i])
            else:
                x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb=-bigM,ub=bigM)
        ### y is post-relu
        y = {}
        for i in range(n_nodes):
            y[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "y[" + str(i) + "]", lb=-bigM, ub = bigM)
        ### z is indicator-relu
        z = {}
        for i in range(n_nodes):
            z[i] = model.addVar(vtype = GRB.BINARY, name = "z[" +str(i) + "]")
            
        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
                
        model.update()

     ### set x from y
        for layer in range(n_layers-1):
            for i in range(layers_size[layer+1]):    
               
                node_index_to_set = offset[layer+1]+i
               
                ### set z from x
                model.addConstr(x[node_index_to_set] <= bigM*z[node_index_to_set] - 1e-10)
                model.addConstr(x[node_index_to_set] >= (-1)*bigM*(1-z[node_index_to_set]))
               
                ### set y from z and x
                model.addConstr(y[node_index_to_set] <= x[node_index_to_set] + bigM*(1-z[node_index_to_set]))
                model.addConstr(y[node_index_to_set] >= x[node_index_to_set] - bigM*(1-z[node_index_to_set]))
               
                model.addConstr(y[node_index_to_set] <= bigM*z[node_index_to_set]- 1e-10)
                model.addConstr(y[node_index_to_set] >= -bigM*z[node_index_to_set])
                
                ### set x from y
                calc_pre_relu_expr = gp.LinExpr(0.0)
                for j in range(n_nodes):
                    calc_pre_relu_expr += y[j]*weight_matrix[j,offset[layer+1]+i]
                calc_pre_relu_expr += + bias_vector[offset[layer+1]+i]
                model.addConstr(calc_pre_relu_expr == x[offset[layer+1] + i])
                ###
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
        
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= beta can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])
        
            
        objective_expr = gp.LinExpr(0.0) 
        objective_expr += x[n_nodes-1]

            
        model.setObjective(objective_expr, objective)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        

        
        nSolutions = model.SolCount

        
        for sol_index in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber,sol_index)

            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
            
            for i in range(n_variables):
                my_sol[0][i] = round(x[i].Xn, 4)

            for i in range(n_nodes):
                if i == n_nodes - 1:
                    my_y.append(x[i].Xn)
                else:
                    my_y.append(y[i].Xn)
             
            my_sols.append(my_sol)
            my_ys.append(my_y)

       
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    except AttributeError:
        print('Encountered an attribute error')    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        return OptimalPredictionUnrestrictedModel, my_sols[0][0], model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'



#%%

def lr_unrestricted(lr_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    BASE model
    """   
    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
    
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
        
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime
    else:
        return 'null', 'null', model.RunTime
    
#%%

def lr_unrestricted_real_data(lr_info, objective = GRB.MAXIMIZE, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    BASE model
    utilized for a real-world dataset
    """   
  
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
    
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    # except AttributeError:
    #     print('Encountered an attribute error')
    
    if nSolutions > 0:
        return model.ObjVal, my_sol, model.RunTime
    else:
        return 'null', 'null', model.RunTime
    
#%%
def lr_mdist(lr_info, true_function, X_train, min_max_scaler, CriteriaMdist, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    MD constraints
    """   
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-Mdist")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
                
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        my_sol = []
        
        for v in model.getVars():
            my_sol.append(v.x)
    
        # print('Obj: %g' % model.objVal)
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPrediction = model.ObjVal 
        TrueValue_OptimalPrediction = true_function(reverse_transform_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime
    else:
        return 'null', 'null', model.RunTime
    
#%%
def lr_mdist_real_data(lr_info, X_train, min_max_scaler, CriteriaMdist, objective = GRB.MAXIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    MD constraints
    utilized for a real-world dataset
    """   
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-Mdist")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
                
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        my_sol = []
        
        for v in model.getVars():
            my_sol.append(v.x)
    
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPrediction = model.ObjVal 
        return OptimalPrediction, my_sol, model.RunTime
    else:
        return 'null', 'null', model.RunTime
      
#%%

def lr_knn(lr_info, true_function, X_train, min_max_scaler, k_neighbors, D, objective = GRB.MINIMIZE, bigM_kNN=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    KNN constraints
    """   
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-KNN")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # kNN relevant variables
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        for j in range(len(LB)):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) 
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)
        num_inputs = len(LB)
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")
        
        # kNN relevant constraints
        for j in range(len(LB)):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(len(LB)):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])

        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        reverse_transform_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = model.ObjVal 
        TrueValue_OptimalPrediction = true_function(reverse_transform_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
    
 
#%%

def lr_knn_real_data(lr_info, X_train, k_neighbors, D, objective = GRB.MAXIMIZE, bigM_kNN=10000.0, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    KNN constraints
    utilized for a real-world dataset
    """   


    X_train = np.array(X_train)    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-KNN")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # kNN relevant variables
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        for j in range(len(LB)):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) 
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)
        num_inputs = len(LB)
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")
        
        # kNN relevant constraints
        for j in range(len(LB)):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(len(LB)):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
        
        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])

        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        return model.ObjVal, my_sol, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
    

#%%

def lr_isolation_forest(lr_info, isolation_forest_info, true_function, X_train, min_max_scaler, L, objective = GRB.MINIMIZE, bigM_if=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    IF constraints
    """   
    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-IsolationForest")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= L can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])

        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        
        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        reverse_transform_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        print(reverse_transform_sol)
        OptimalPrediction = model.ObjVal
        TrueValue_OptimalPrediction = true_function(reverse_transform_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime, model.MIPGap
    else:
        return 'null', 'null',model.RunTime, 'null'


#%%

def lr_isolation_forest_real_data(lr_info, isolation_forest_info, X_train, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    IF constraints
    utilized for a real-world dataset
    """   
    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-IsolationForest")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= L can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])

        # Set objective
        obj = gp.LinExpr(0.0)
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        
        
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        for sol_index in range(1):
            model.setParam(GRB.Param.SolutionNumber,sol_index)
            my_sol_midpoint = {}
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
    if nSolutions > 0:
        reverse_transform_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        print(reverse_transform_sol)
        OptimalPrediction = model.ObjVal
        return OptimalPrediction, my_sol, model.RunTime, model.MIPGap
    else:
        return 'null', 'null',model.RunTime, 'null'



#%%

def rf_unrestricted(rf_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    BASE model
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Consistency_Model")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])

    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
    
            
        obj = gp.LinExpr(0.0) # QuadExpr
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
        
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:  
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round((my_sol_LB[i] + my_sol_UB[i]) / 2, 4)       

                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal - lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))


    reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
    OptimalPredictionUnrestrictedModel = opt_obj 
    TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
    return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap

#%%

def rf_unrestricted_real_data(rf_info, objective = GRB.MAXIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    BASE model
    utilized for a real-world dataset
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Consistency_Model")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])

    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
    
            
        obj = gp.LinExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
        
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:  
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round((my_sol_LB[i] + my_sol_UB[i]) / 2, 4)       

                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
                print ('obj approach 2: ', obj_rf / len(trees))
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))

    OptimalPredictionUnrestrictedModel = opt_obj 
    return OptimalPredictionUnrestrictedModel, my_sol, model.RunTime, model.MIPGap


#%%

def rf_mdist(rf_info, true_function, X_train, min_max_scaler, CriteriaMdist, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    MD constraints
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_M_distance")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])



        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce MD on center of the optimal intervals

        
        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
        
            
        obj = gp.LinExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:      
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)   

                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))


    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null',model.RunTime, 'null'

#%%

def rf_mdist_real_data(rf_info, X_train, min_max_scaler, CriteriaMdist, objective = GRB.MAXIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    MD constraints
    utilized for a real-world dataset
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_M_distance")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])



        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce MD on center of the optimal intervals

        # add constraint on M-distance of x: Mdist(x) <= k    
        M_distance_temp = MDistance_Squared_SinglePoint(sol_temp, X_train)
        model.addConstr(M_distance_temp <= CriteriaMdist)
        
            
        obj = gp.LinExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:      
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)   

                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))


    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        return OptimalPredictionUnrestrictedModel, my_sol, model.RunTime, model.MIPGap
    else:
        return 'null', 'null',model.RunTime, 'null'

#%%

def rf_isolation_forest(rf_info, isolation_forest_info, true_function, X_train, min_max_scaler, L, objective = GRB.MINIMIZE, bigM_if=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    IF constraints
    """  
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_Isolation_Forest")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])

        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce IF on center of the optimal intervals

    
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= L can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])
        
            
        obj = gp.LinExpr(0.0) # QuadExpr
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)

    
                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
                print ('obj approach 2: ', obj_rf / len(trees))
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    except AttributeError:
        print('Encountered an attribute error')

    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'

#%%

def rf_isolation_forest_real_data(rf_info, isolation_forest_info, X_train, min_max_scaler, L, objective = GRB.MAXIMIZE, bigM_if=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    IF constraints
    utilized for a real-world dataset
    """  
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    IF_trees = isolation_forest_info['trees']
    IF_leafnodes = isolation_forest_info['leafnodes']
    IF_splitnodes = isolation_forest_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_Isolation_Forest")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])

        # isolation forest relevant variables
        IF_y = {}
        for i in range(len(IF_trees)):
            IF_y[i] = {}
            for j in range(len(IF_leafnodes[i])):
                IF_y[i][j] = model.addVar(vtype = GRB.BINARY, name = "l[" +str(i) + "]" + "[" + str(j) + "]")

        IF_z_LB = {}
        for i in range(len(LB)):
            IF_z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        IF_z_UB = {}
        for i in range(len(LB)):
            IF_z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) 

    
        # isolation forest relevant constraints
        # choose one leafnode per tree
        for i in range(len(IF_trees)):
            equ = gp.LinExpr(0.0)
            for j in range(len(IF_leafnodes[i])):
                equ += IF_y[i][j]
            model.addConstr(equ == 1)
            
        # leafnode with depth <= L can't be chosen        
        for i in range(len(IF_trees)):
            for j in range(len(IF_leafnodes[i])):
                if len(IF_leafnodes[i][j].splitnode_ids) <= L:
                    model.addConstr(IF_y[i][j] == 0)
        
        # feasible region of chosen leafnode
        for t in range(len(IF_trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(IF_leafnodes[t])):
                    equLB += IF_leafnodes[t][l].lb[f] * IF_y[t][l]
                    equUB += (1 - IF_leafnodes[t][l].ub[f]) * IF_y[t][l]
                model.addConstr(equLB <= IF_z_LB[f])
                model.addConstr(1 - equUB >= IF_z_UB[f])
    
        # overlapped region should be nonempty
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= IF_z_UB[f] - 1e-4)
            
        # feasible region of decision variable of x should be updated according to the overlapped region
        for f in range(len(LB)):
            model.addConstr(IF_z_LB[f] <= x[f])
            model.addConstr(IF_z_UB[f] >= x[f])
        
            
        obj = gp.LinExpr(0.0) # QuadExpr
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)

    
                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
                print ('obj approach 2: ', obj_rf / len(trees))
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))
    except AttributeError:
        print('Encountered an attribute error')

    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        return OptimalPredictionUnrestrictedModel, my_sol, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'

#%%

def rf_kNN(true_function, min_max_scaler, X_train, k_neighbors, D, rf_info, objective = GRB.MINIMIZE, bigM_kNN=10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    KNN constraints
    """      
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_M_distance")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])

    # kNN relevant variables
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        for j in range(len(LB)):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) 
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)
        num_inputs = len(LB)
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce KNN on center of the optimal intervals

        
        # kNN relevant constraints
        for j in range(len(LB)):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])              
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(len(LB)):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
        
            
        obj = gp.LinExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:  
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)

    
                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
    
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
                print ('obj approach 2: ', obj_rf / len(trees))
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))


    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null',model.RunTime, 'null'

#%%

def rf_kNN_real_data(X_train, k_neighbors, D, rf_info, objective = GRB.MAXIMIZE, bigM_kNN=10000.0, OutputFlag = 0, timelimit = 100): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    KNN constraints
    utilized for a real-world dataset
    """  
    
    X_train = np.array(X_train)
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
    
    try:
       
        model = gp.Model("RF_Opt_Model_M_distance")
    
    # decision variables
        y = {}
        for i in range(len(trees)):
            y[i] = {}
            for j in range(len(leafnodes[i])):
                y[i][j] = model.addVar(vtype = GRB.BINARY, name = "y[" +str(i) + "]" + "[" + str(j) + "]")
        z_LB = {}
        for i in range(len(LB)):
            z_LB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_LB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        z_UB = {}
        for i in range(len(LB)):
            z_UB[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "z_UB[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "x[" + str(i) + "]", lb = LB[i], ub = UB[i], obj = 0.0)
            sol_temp.append(x[i])

    # kNN relevant variables
        aux_pos = {}
        aux_neg = {}
        w = {}
        d = {}
        p = {}
        
        for j in range(len(LB)):
            aux_pos[j] = {}
            aux_neg[j] = {}
            for i in range(len(X_train)):
                aux_pos[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_pos[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0) 
                aux_neg[j][i] = model.addVar(vtype = GRB.CONTINUOUS, name = "aux_neg[" + str(j) + "][" + str(i) + "]", lb = 0.0, ub = 1.0)
        num_inputs = len(LB)
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)

        for k in range(k_neighbors):
            d[k] = model.addVar(vtype = GRB.CONTINUOUS, name = "d[" + str(i) + "]", lb = 0.0, ub = 2*num_inputs)
            p[k] = {}
            for i in range(len(X_train)):
                p[k][i] = model.addVar(vtype = GRB.BINARY, name = "pi[" + str(k) + "][" + str(i) + "]")

        model.update()
        
    
    # add constraints
        for t in range(len(trees)):
            equ = gp.LinExpr(0.0)
            for l in range(len(leafnodes[t])):
                equ += y[t][l]
            model.addConstr(equ == 1)
    
        for t in range(len(trees)):
            for f in range(len(LB)):
                equLB = gp.LinExpr(0.0)
                equUB = gp.LinExpr(0.0)
                for l in range(len(leafnodes[t])):
                    equLB += leafnodes[t][l].lb[f] * y[t][l]
                    equUB += (1 - leafnodes[t][l].ub[f]) * y[t][l]
                model.addConstr(equLB <= z_LB[f])
                model.addConstr(1 - equUB >= z_UB[f])
    
        for f in range(len(LB)):
            model.addConstr(z_LB[f] <= z_UB[f] - 1e-4)
        
        for f in range(len(LB)):
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce KNN on center of the optimal intervals

        
        # kNN relevant constraints
        for j in range(len(LB)):
            for i in range(len(X_train)):
                model.addConstr(x[j] - aux_pos[j][i] + aux_neg[j][i] == X_train[i][j])
                
                
        for i in range(len(X_train)):
            dist = 0
            for j in range(len(LB)):
                dist = dist + aux_pos[j][i] + aux_neg[j][i]
            model.addConstr(w[i] == dist)
            
        for k in range(k_neighbors):
            for i in range(len(X_train)):             
                model.addConstr(d[k] >= w[i] - bigM_kNN*(1 - p[k][i]))

        for k in range(k_neighbors):
            p_temp = 0
            for i in range(len(X_train)):
                p_temp += p[k][i]
            model.addConstr(p_temp == k + 1)
            
        avg_dist = 0
        for k in range(k_neighbors):
            avg_dist += d[k]
        model.addConstr(avg_dist <= D * k_neighbors)
        
            
        obj = gp.LinExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:  
            for sol_index in range(1):
                model.setParam(GRB.Param.SolutionNumber,sol_index)
        
                my_sol_LB = {} 
                my_sol_UB = {}
                my_sol_midpoint = {}						           
                for i in range(len(LB)):
                    my_sol_LB[i] = LB[i]
                    my_sol_UB[i] = UB[i]
                    for t in range(len(trees)):
                        for l in range(len(leafnodes[t])):
                            if y[t][l].Xn > 0.5:
                                my_sol_LB[i] = max(my_sol_LB[i], leafnodes[t][l].lb[i])
                                my_sol_UB[i] = min(my_sol_UB[i], leafnodes[t][l].ub[i])
                                
                    my_sol_midpoint[i] = round(x[i].Xn, 4)

    
                my_sol = list(my_sol_midpoint.values())
                my_sol_chosen_leafnodes = {}
    
                lbub = 0
                for f in range(len(LB)):
                    lbub += z_LB[f].Xn - z_UB[f].Xn
                opt_obj = round((model.ObjVal-lbub) / len(trees), 4)
                print ('obj approach 1: ', opt_obj)
                
                obj_rf = 0
                for t in range(len(trees)):
                    my_sol_chosen_leafnodes[t] = {}

                    for l in range(len(leafnodes[t])):
                        my_sol_chosen_leafnodes[t][l] = y[t][l].Xn
                        if my_sol_chosen_leafnodes[t][l] > 0.5:
                            obj_rf += leafnodes[t][l].value
                opt_obj = obj_rf / len(trees)
                print ('obj approach 2: ', obj_rf / len(trees))
    except gp.GurobiError as e:
        print('Gurobi error ' + str(e.errno) + ": " + str(e.message))


    return opt_obj, my_sol, model.RunTime, model.MIPGap


