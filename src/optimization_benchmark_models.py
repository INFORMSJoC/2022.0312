#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp

import gurobipy as gp
from gurobipy import GRB

from pca_distance_calculator import PCA_Distance_SinglePoint
from sklearn.preprocessing import StandardScaler  
standard_scaler = StandardScaler()
from sklearn.svm import OneClassSVM


#%%
# SVM-BC: Branch-and-Cut Heuristic (See Algorithm 1 in the paper)
def mycallback(model, where):
    rho = model._rho
    sv = model._sv
    alpha = model._alpha
    gamma = model._gamma
    n_variables = model._n_variables
    n_pieces = model._n_pieces  
    if where == GRB.Callback.MIPSOL:
        # print('===========================GRB.MIPSOL=======================')
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        #print("==============================objective: {}".format(obj))
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(model._x)
        x_discrete = {}
        for i in range(n_variables):
            x_discrete[i] = model.cbGetSolution(model._x_discrete[i])

        # add lazy cut if the solution violate the decision function of SVM
        d = {}
        for i in range(len(sv)):
            d[i] = 0
            for j in range(n_variables):
                d[i] += (sv[i][j] - x[j]) * (sv[i][j] - x[j])
        k = {}
        for i in range(len(sv)):
            k[i] = np.exp(-gamma*d[i])
        svm_df = 0
        for i in range(len(sv)):
            svm_df += alpha[i]*k[i]
        if round(svm_df, 5) < round(rho, 5):
            # print(svm_df, rho)
            #print('******************cut added*******************************')
            # add x segments elimination constr
            equ = gp.LinExpr(0.0)
            for i in range(n_variables):
                for p in range(n_pieces):
                    if x_discrete[i][p] > 0.5:
                        equ += model._x_discrete[i][p]
            model.cbLazy(equ <= n_variables - 1)

#%%

def nn_convex_hull(nn, true_function, X_train, LB, UB, min_max_scaler, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    CH constraints
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
                
        # convex hull relevant variables
        w = {}
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = GRB.INFINITY)
        
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
       
        # convex hull relevant constraints                
        for i in range(len(LB)):
            equ = gp.LinExpr(0.0)
            for j in range(len(X_train)):
                equ += w[j]*X_train[j][i]
            model.addConstr(equ == x[i])
        
        equ = gp.LinExpr(0.0)
        for i in range(len(X_train)):
            equ += w[i]
        model.addConstr(equ == 1)
            
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

        
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)

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
    except AttributeError:
        print('Encountered an attribute error')    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sol[0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', 'null', 'null'


# In[111]:

def nn_pca(nn_info, true_function, X_train, LB, UB, min_max_scaler, lambda_penalty_parameter, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    PCA constraints converted to penalty term in the objective function
    """
    
    weight_matrix = nn_info['Ws']
    bias_vector = nn_info['bs']
    layers_size = nn_info['layers_size']
    n_nodes = nn_info['n_nodes']
    offset = nn_info['offset']
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
                ###
        for i in range(num_inputs):
            model.addConstr(x[i] == y[i])
            model.addConstr(z[i] == 0)
      
            
        objective_expr = gp.QuadExpr(0.0) 
        objective_expr += x[n_nodes-1]
        objective_expr.add(PCA_Distance_SinglePoint(sol_temp, X_train), lambda_penalty_parameter)

        
        model.setObjective(objective_expr, objective)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        

        
        nSolutions = model.SolCount

        
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)

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
    except AttributeError:
        print('Encountered an attribute error')    

    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol[0]])[0]
        OptimalPrediction = model.ObjVal - PCA_Distance_SinglePoint(my_sol[0], X_train)*lambda_penalty_parameter
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
# In[111]:

def nn_pca_constr(nn_info, true_function, X_train, LB, UB, min_max_scaler, lambda_criteria, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    PCA constraints
    """
    
    weight_matrix = nn_info['Ws']
    bias_vector = nn_info['bs']
    layers_size = nn_info['layers_size']
    n_nodes = nn_info['n_nodes']
    offset = nn_info['offset']
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
        PCA_distance_temp = PCA_Distance_SinglePoint(sol_temp, X_train)
        model.addConstr(PCA_distance_temp <= lambda_criteria) 
            
        objective_expr = gp.LinExpr(0.0)
        objective_expr += x[n_nodes-1]

        
        model.setObjective(objective_expr, objective)
       	model.setParam('OutputFlag', OutputFlag)
        model.setParam('TimeLimit', timelimit)   
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        

        
        nSolutions = model.SolCount

        
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)

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
    except AttributeError:
        print('Encountered an attribute error')    

    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol[0]])[0]
        OptimalPrediction = model.ObjVal
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
#%%

def nn_svm_w_callback(svm, rho_new, nn, true_function, X_train, LB, UB, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, bigM = 10000.0, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    SVM-BC constraints
    """
    
    bigM = 10000
    weight_matrix = nn['Ws']
    bias_vector = nn['bs']
    
    layers_size = nn['layers_size']
    n_nodes = nn['n_nodes']
    offset = nn['offset']
    n_layers = len(layers_size)
    num_inputs = layers_size[0]
    
    clf = svm
    
    my_sols = []
    my_ys = []
    
        
    try:        
        model = gp.Model("base_ip")
        ### data statements    
        model._rho = rho_new  
        model._sv = clf.support_vectors_.tolist()
        model._alpha = clf.dual_coef_.ravel().tolist()
        model._gamma = clf._gamma
        model._n_variables = num_inputs
        model._n_pieces = x_segment_count
    
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
    
        # discreted domain segment for each feature
        x_discrete = {}
        for i in range(model._n_variables):
            x_discrete[i] = {}
            for p in range(model._n_pieces):
                x_discrete[i][p] = model.addVar(vtype = GRB.BINARY, name = "x_discrete[" +str(i) + "]" + "[" + str(p) + "]")
    
        model.update()
    
        ### set x from x_discrete
        for i in range(model._n_variables):
            equ = gp.LinExpr(0.0)
            for p in range(model._n_pieces): 
                equ += x_discrete[i][p]
                if p == 0:
                    model.addConstr(x[i] >= LB[i] * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
                else:
                    model.addConstr(x[i] >= (LB[i] + p / model._n_pieces + 0.1 / model._n_pieces) * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
            model.addConstr(equ == 1)

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
                 
        objective_expr = gp.LinExpr(0.0)
        objective_expr += x[n_nodes-1]
        
        model.setObjective(objective_expr, objective)
        model.setParam('OutputFlag', OutputFlag)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)  
        model.setParam('Threads', 1)
        model.setParam('LazyConstraints', 1)

        model.update()
        model._x = x
        model._x_discrete = x_discrete
        model.optimize(mycallback)
        
    
        
        nSolutions = model.SolCount
    
        
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
    
            my_sol=np.zeros(shape=(1,num_inputs))
            my_y = []
            
            for i in range(num_inputs):
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
    except AttributeError:
        print('Encountered an attribute error')
    
    if nSolutions > 0:
        reverse_transform_sol = min_max_scaler.inverse_transform([my_sols[0][0]])[0]
        OptimalPredictionUnrestrictedModel = model.ObjVal 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_transform_sol)
        print(my_sols[0][0])
        print('optimal obj in scale of [0, 1]: ', OptimalPredictionUnrestrictedModel)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'
#%%

def lr_convex_hull(lr_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    CH constraints
    """  
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-ConvexHull")
    
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # convex hull relevant variables
        w = {}
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = GRB.INFINITY)
        
        # convex hull relevant constraints                
        for i in range(len(LB)):
            equ = gp.LinExpr(0.0)
            for j in range(len(X_train)):
                equ += w[j]*X_train[j][i]
            model.addConstr(equ == x[i])
        
        equ = gp.LinExpr(0.0)
        for i in range(len(X_train)):
            equ += w[i]
        model.addConstr(equ == 1)
        
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
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
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
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime
    else:
        return 'null', 'null','null'


#%%

def lr_pca(lr_info, true_function, X_train, min_max_scaler, lambda_penalty_parameter, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    PCA constraints converted to penalty term in the objective function
    """  
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    # X_train_copy = X_train.copy()
    # X_train_standardized = standard_scaler.fit_transform(X_train_copy)

    try:

        # Create a new model
        model = gp.Model("LinearRegression-PCA")
        print("lambda_penalty_parameter: " + str(lambda_penalty_parameter))
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
       
        # Set objective
        obj = gp.QuadExpr()  
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        obj.add(PCA_Distance_SinglePoint(sol_temp, X_train), lambda_penalty_parameter)           
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
            my_sol_midpoint = {}
            opt_obj = intercept
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
                opt_obj += coef[i] * x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPrediction = model.ObjVal-PCA_Distance_SinglePoint(my_sol, X_train)*lambda_penalty_parameter
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime 
    else:
        return 'null', 'null', model.RunTime

#%%

def lr_pca_constr(lr_info, true_function, X_train, min_max_scaler, lambda_criteria, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    PCA constraints
    """    
      
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']
    
    try:

        # Create a new model
        model = gp.Model("LinearRegression-PCA")
        print("lambda_criteria: " + str(lambda_criteria))
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        PCA_distance_temp = PCA_Distance_SinglePoint(sol_temp, X_train)
        model.addConstr(PCA_distance_temp <= lambda_criteria) 
       
        # Set objective
        obj = gp.LinExpr()  
        obj.addConstant(intercept)
        for i in range(len(LB)):
            obj.addTerms(coef[i], x[i])
        model.setObjective(obj, objective)
    
        # Optimize model
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('TimeLimit', timelimit)
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.optimize()
        nSolutions = model.SolCount
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
            my_sol_midpoint = {}
            opt_obj = intercept
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x
                opt_obj += coef[i] * x[i].x
            my_sol = list(my_sol_midpoint.values())
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPrediction = model.ObjVal
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime
    else:
        return 'null', 'null', model.RunTime


#%%

def lr_svm_w_callback(svm, rho_new, lr_info, true_function, X_train, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, OutputFlag = 0, timelimit = 1800): 
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    SVM-BC constraints
    """  
        
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']

    clf = svm

    try:

        # Create a new model
        model = gp.Model("LinearRegressionBasedModel-SVM")
        ### data statements    
        model._rho = rho_new
        model._sv = clf.support_vectors_.tolist()
        model._alpha = clf.dual_coef_.ravel().tolist()
        model._gamma = clf._gamma
        model._n_variables = len(LB)
        model._n_pieces = x_segment_count
        
        # Create variables
        x = {}
        sol_temp = []
        for i in range(len(LB)):
            x[i] = model.addVar(lb = LB[i], ub = UB[i], vtype = GRB.CONTINUOUS, name="x[%s]"%i)
            sol_temp.append(x[i])
        
        # discreted domain segment for each feature
        x_discrete = {}
        for i in range(model._n_variables):
            x_discrete[i] = {}
            for p in range(model._n_pieces):
                x_discrete[i][p] = model.addVar(vtype = GRB.BINARY, name = "x_discrete[" +str(i) + "]" + "[" + str(p) + "]")
    
        model.update()
    
        ### set x from x_discrete
        # model.addConstr(x)
        
        for i in range(model._n_variables):
            equ = gp.LinExpr(0.0)
            for p in range(model._n_pieces): 
                equ += x_discrete[i][p]
                if p == 0:
                    model.addConstr(x[i] >= LB[i] * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
                else:
                    model.addConstr(x[i] >= (LB[i] + p / model._n_pieces + 0.1 / model._n_pieces) * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
            model.addConstr(equ == 1)

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
        model.setParam('LazyConstraints', 1)
    

        model.update()
        model._x = x
        model._x_discrete = x_discrete
        model.optimize(mycallback)
        
        nSolutions = model.SolCount
        print("nSolutions: " + str(nSolutions))
        

        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
            my_sol_midpoint = {}
            for i in range(len(LB)):
                my_sol_midpoint[i] = x[i].x

            my_sol = list(my_sol_midpoint.values())

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    if nSolutions > 0:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = model.ObjVal 
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        return OptimalPrediction, TrueValue_OptimalPrediction, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime,'null'
#%%

def rf_convex_hull(rf_info, true_function, X_train, min_max_scaler, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """optimize individual RF instance
    rf_info is a dictionary storing rf info
    opt model: objective is to find input x's to maximize / minimize the output of RF 
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
        # convex hull relevant variables
        w = {}
        for i in range(len(X_train)):
            w[i] = model.addVar(vtype = GRB.CONTINUOUS, name = "w[" + str(i) + "]", lb = 0.0, ub = GRB.INFINITY)
        
        # convex hull relevant constraints                
        for i in range(len(LB)):
            equ = gp.LinExpr(0.0)
            for j in range(len(X_train)):
                equ += w[j]*X_train[j][i]
            model.addConstr(equ == x[i])
        
        equ = gp.LinExpr(0.0)
        for i in range(len(X_train)):
            equ += w[i]
        model.addConstr(equ == 1)      
    
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
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce CH on center of the optimal intervals

                
            
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
            model.setParam(GRB.Param.SolutionNumber, 0)
    
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

def rf_pca(rf_info, true_function, X_train, min_max_scaler, lambda_penalty_parameter, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    PCA constraints converted to penalty term in the objective function
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
            
    try:
       
        model = gp.Model("RF_Opt_Model_PCA")
    
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
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce PCA on center of the optimal intervals

                
            
        obj = gp.QuadExpr(0.0) 
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                obj += leafnodes[t][l].value * y[t][l]
        
        for f in range(len(LB)):  
            obj += z_LB[f] - z_UB[f]
         
        obj.add(PCA_Distance_SinglePoint(sol_temp, X_train), lambda_penalty_parameter)           
            
        model.setObjective(obj, objective)
        model.setParam('OutputFlag', OutputFlag);
        model.setParam('MIPGapAbs', 1e-5)
        model.setParam('TimeLimit', timelimit)
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
    
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
    except AttributeError:
        print('Encountered an attribute error')

    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_minmaxed_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
           
    else:
        return 'null', 'null', model.RunTime, 'null'
 
    
#%%

def rf_pca_constr(rf_info, true_function, X_train, min_max_scaler, lambda_criteria, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    PCA constraints
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
            
    try:
       
        model = gp.Model("RF_Opt_Model_PCA")
    
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
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce PCA on center of the optimal intervals

        
        PCA_distance_temp = PCA_Distance_SinglePoint(sol_temp, X_train)
        model.addConstr(PCA_distance_temp <= lambda_criteria)           
               
            
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
        model.setParam('NonConvex', 2)
        model.setParam('Threads', 1)
        model.Params.PoolSearchMode = 0
        # only store one solution
        model.Params.PoolSolutions = 1
        
        model.update()
        
        model.optimize()
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
    
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
    except AttributeError:
        print('Encountered an attribute error')
    
    if nSolutions > 0:
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_minmaxed_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
           
    else:
        return 'null', 'null', model.RunTime, 'null'
 
    

#%%

def rf_svm_w_callback(svm, rho_new, rf_info, true_function, X_train, min_max_scaler, x_segment_count, objective = GRB.MINIMIZE, OutputFlag = 1, timelimit = 1800): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    SVM-BC constraints
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
        
    clf = svm
    
    try:
       
        model = gp.Model("RF_Opt_Model_SVM")

        ### data statements    
        model._rho = rho_new
        model._sv = clf.support_vectors_.tolist()
        model._alpha = clf.dual_coef_.ravel().tolist()
        model._gamma = clf._gamma
        model._n_variables = len(LB)
        model._n_pieces = x_segment_count
    
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

        # discreted domain segment for each feature
        x_discrete = {}
        for i in range(model._n_variables):
            x_discrete[i] = {}
            for p in range(model._n_pieces):
                x_discrete[i][p] = model.addVar(vtype = GRB.BINARY, name = "x_discrete[" +str(i) + "]" + "[" + str(p) + "]")
    
        model.update()
        ### set x from x_discrete
        for i in range(model._n_variables):
            equ = gp.LinExpr(0.0)
            for p in range(model._n_pieces): 
                equ += x_discrete[i][p]
                if p == 0:
                    model.addConstr(x[i] >= LB[i] * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
                else:
                    model.addConstr(x[i] >= (LB[i] + p / model._n_pieces + 0.1 / model._n_pieces) * x_discrete[i][p])
                    model.addConstr(x[i] <= UB[i] * (p+1) / model._n_pieces + 1 - x_discrete[i][p])
            model.addConstr(equ == 1)
    
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
            model.addConstr(x[f] * 2 == z_LB[f] + z_UB[f]) # enforce SVM on center of the optimal intervals

         

        # set objective    
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
        model.setParam('LazyConstraints', 1)

        
        model.update()
        model._x = x
        model._x_discrete = x_discrete
        model.optimize(mycallback)
        
        nSolutions = model.SolCount
    # store solutions
        if nSolutions > 0:
            model.setParam(GRB.Param.SolutionNumber, 0)
    
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
        reverse_minmaxed_sol = min_max_scaler.inverse_transform([my_sol])[0]
        OptimalPredictionUnrestrictedModel = opt_obj 
        TrueValue_OptimalPredictionUnrestrictedModel = true_function(reverse_minmaxed_sol)
        return OptimalPredictionUnrestrictedModel, TrueValue_OptimalPredictionUnrestrictedModel, model.RunTime, model.MIPGap
    else:
        return 'null', 'null', model.RunTime, 'null'