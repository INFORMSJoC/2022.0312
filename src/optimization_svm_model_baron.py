#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.preprocessing import StandardScaler  
standard_scaler = StandardScaler()
from sklearn.svm import OneClassSVM

from pyomo.environ import *
from pyomo.opt import SolverFactory

#%%

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

#%%

def Baron_MIPGap():
    """
    Calculate MIPGap by reading output from Baron 
    """
    f = open("baron_result.txt", "r") # baron_result
    content = f.read()
    content_list = content.splitlines()
    f.close()
    
    for i in range(len(content_list)):
        single_line = content_list[i]
        single_line_join = ''.join(single_line)
        single_line_join_filtered = list(filter(None, single_line_join.split('  ')))
        if len(single_line_join_filtered) != 0 and single_line_join_filtered[0] == ' Best solution found at node:':
            if single_line_join_filtered[1] == '-3':
                solution_status = 'no feasible solution'
                MIPGap = 'null'
            elif single_line_join_filtered[1] == '-2': 
                solution_status = 'user-supplied best solution'
                MIPGap = 0
            elif single_line_join_filtered[1] == '-1': 
                solution_status = 'best solution found in preprocessing'
                MIPGap = 0
            else:
                solution_status = 'best solution found in one node of the tree'
    
    if solution_status == 'best solution found in one node of the tree':
        for i in range(len(content_list)):
            single_line = content_list[i]
            single_line_join = ''.join(single_line)
            single_line_join_filtered = list(filter(None, single_line_join.split('  ')))
            if single_line_join_filtered == ['Iteration', 'Open nodes', ' Time (s)', 'Lower bound', 'Upper bound']:
                index_Iteration = content_list.index(single_line)

    
        for i in range(index_Iteration, len(content_list)):  
            single_line = content_list[i]
            single_line_join = ''.join(single_line)
            single_line_join_filtered = list(filter(None, single_line_join.split('  ')))
            

            if len(single_line_join_filtered) == 0:
                line_lb_ub = content_list[i-1]
                line_lb_ub_join = ''.join(line_lb_ub)
                line_lb_ub_join_filtered = list(filter(None, line_lb_ub_join.split('  ')))
                LB = float(line_lb_ub_join_filtered[-2])
                UB = float(line_lb_ub_join_filtered[-1])
                if UB != 0:
                    MIPGap = abs((UB-LB)/UB)
                else:
                    MIPGap = 'null'
                break
        
    return MIPGap


#%%

def nn_svm_baron_rho_new(svm, nn_info, rho_new, true_function, X_train, LB, UB, min_max_scaler, objective = minimize, bigM = 10000.0, timelimit = 1800, starting_point = None):
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    SVM constraints
    """
    
    weight_matrix = nn_info['Ws']
    bias_vector = nn_info['bs']
    layers_size = nn_info['layers_size']
    n_nodes = nn_info['n_nodes']
    offset = nn_info['offset']
    n_layers = len(layers_size)    
    num_inputs = layers_size[0]
    my_sols = []
    my_ys = []
    
    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma
       
    # Create a new model
    model = ConcreteModel("PMO(NN)-SVM_baron w/rho_new")
    
    # Create variables
    model.features_dim = range(num_inputs)
    ### x is pre-relu
    model.n_nodes = range(n_nodes)
    x_lb = [0.0] * num_inputs + [-bigM] * (n_nodes - num_inputs)
    x_ub = [1.0] * num_inputs + [bigM] * (n_nodes - num_inputs)
    def x_bound(model,i):
        return (x_lb[i], x_ub[i])
    model.x = Var(model.n_nodes, bounds = x_bound)
    ### y is post-relu            
    model.y = Var(model.n_nodes, bounds = (-bigM, bigM))  
    ### z is indicator-relu
    model.z = Var(model.n_nodes, within = Binary)         

    ### svm relevant variables
    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))

    if starting_point != None:
        for i in range(n_nodes):
            model.x[i] = starting_point['x'][i]
            model.y[i] = starting_point['y'][i]    
            model.z[i] = starting_point['z'][i]
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]
    # Add constraints        
    ### set x from y      
    model.z_from_x = ConstraintList()
    model.y_from_zx = ConstraintList()
    model.x_from_y = ConstraintList()
    for layer in range(n_layers-1):
        for i in range(layers_size[layer+1]): 
            node_index_to_set = offset[layer+1]+i
            ### set z from x
            model.z_from_x.add(expr = model.x[node_index_to_set] <= bigM * model.z[node_index_to_set] - 1e-10)
            model.z_from_x.add(expr = model.x[node_index_to_set] >= (-1) * bigM * (1 - model.z[node_index_to_set]))
           
            ### set y from z and x
            model.y_from_zx.add(expr = model.y[node_index_to_set] <= model.x[node_index_to_set] + bigM * (1 - model.z[node_index_to_set]))
            model.y_from_zx.add(expr = model.y[node_index_to_set] >= model.x[node_index_to_set] - bigM * (1 - model.z[node_index_to_set]))
            model.y_from_zx.add(expr = model.y[node_index_to_set] <= bigM * model.z[node_index_to_set] - 1e-10)
            model.y_from_zx.add(expr = model.y[node_index_to_set] >= - bigM * model.z[node_index_to_set])
            ### set x from y
            model.x_from_y.add(expr = bias_vector[offset[layer+1]+i] + sum(model.y[j] * weight_matrix[j,offset[layer+1]+i] for j in model.n_nodes) == model.x[offset[layer+1] + i])
  
    model.inputs_constr_x = ConstraintList()
    for i in range(num_inputs):
        model.inputs_constr_x.add(expr = model.x[i] == model.y[i])
    model.inputs_constr_z = ConstraintList()
    for i in range(num_inputs):
        model.inputs_constr_z.add(expr = model.z[i] == 0)       
    
    ### SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho) 
    
    # Set objective
    model.obj = Objective(expr = model.x[n_nodes - 1], sense = objective)
    
    # Optimize model
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")
           
    my_sol = []
    for i in model.x: 
        if i < num_inputs:
            my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = model.x[n_nodes - 1].value
        print(results.solver.termination_condition)
        print(my_sol)
        print('optimal obj in scale of [0, 1]: ', OptimalPrediction)
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = 'null'
        TrueValue_OptimalPrediction = results.solver.termination_condition
        runtime = results.solver.time        
    return OptimalPrediction, TrueValue_OptimalPrediction, runtime

#%%

def nn_svm_baron_rho_new_real_data(svm, nn_info, rho_new, X_train, LB, UB, min_max_scaler, objective = maximize, bigM = 10000.0, timelimit = 1800, starting_point = None):
    """
    OPPM (Neural Network), nn is a dictionary storing information of a pre-trained neural network model
    SVM constraints
    utilized for a real-world dataset
    """
    
    weight_matrix = nn_info['Ws']
    bias_vector = nn_info['bs']
    layers_size = nn_info['layers_size']
    n_nodes = nn_info['n_nodes']
    offset = nn_info['offset']
    n_layers = len(layers_size)    
    num_inputs = layers_size[0]
    my_sols = []
    my_ys = []
    
    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma
       
    # Create a new model
    model = ConcreteModel("PMO(NN)-SVM_baron w/rho_new")
    
    # Create variables
    model.features_dim = range(num_inputs)
    ### x is pre-relu
    model.n_nodes = range(n_nodes)
    x_lb = [0.0] * num_inputs + [-bigM] * (n_nodes - num_inputs)
    x_ub = [1.0] * num_inputs + [bigM] * (n_nodes - num_inputs)
    def x_bound(model,i):
        return (x_lb[i], x_ub[i])
    model.x = Var(model.n_nodes, bounds = x_bound)
    ### y is post-relu            
    model.y = Var(model.n_nodes, bounds = (-bigM, bigM))  
    ### z is indicator-relu
    model.z = Var(model.n_nodes, within = Binary)         

    ### svm relevant variables
    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))

    if starting_point != None:
        for i in range(n_nodes):
            model.x[i] = starting_point['x'][i]
            model.y[i] = starting_point['y'][i]    
            model.z[i] = starting_point['z'][i]
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]
    # Add constraints        
    ### set x from y      
    model.z_from_x = ConstraintList()
    model.y_from_zx = ConstraintList()
    model.x_from_y = ConstraintList()
    for layer in range(n_layers-1):
        for i in range(layers_size[layer+1]): 
            node_index_to_set = offset[layer+1]+i
            ### set z from x
            model.z_from_x.add(expr = model.x[node_index_to_set] <= bigM * model.z[node_index_to_set] - 1e-10)
            model.z_from_x.add(expr = model.x[node_index_to_set] >= (-1) * bigM * (1 - model.z[node_index_to_set]))
           
            ### set y from z and x
            model.y_from_zx.add(expr = model.y[node_index_to_set] <= model.x[node_index_to_set] + bigM * (1 - model.z[node_index_to_set]))
            model.y_from_zx.add(expr = model.y[node_index_to_set] >= model.x[node_index_to_set] - bigM * (1 - model.z[node_index_to_set]))
            model.y_from_zx.add(expr = model.y[node_index_to_set] <= bigM * model.z[node_index_to_set] - 1e-10)
            model.y_from_zx.add(expr = model.y[node_index_to_set] >= - bigM * model.z[node_index_to_set])
            ### set x from y
            model.x_from_y.add(expr = bias_vector[offset[layer+1]+i] + sum(model.y[j] * weight_matrix[j,offset[layer+1]+i] for j in model.n_nodes) == model.x[offset[layer+1] + i])
  
    model.inputs_constr_x = ConstraintList()
    for i in range(num_inputs):
        model.inputs_constr_x.add(expr = model.x[i] == model.y[i])
    model.inputs_constr_z = ConstraintList()
    for i in range(num_inputs):
        model.inputs_constr_z.add(expr = model.z[i] == 0)       
    
    ### SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho) 
    
    # Set objective
    model.obj = Objective(expr = model.x[n_nodes - 1], sense = objective)
    
    # Optimize model
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")
           
    my_sol = []
    for i in model.x: 
        if i < num_inputs:
            my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = model.x[n_nodes - 1].value
        print(results.solver.termination_condition)
        print(my_sol)
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = results.solver.termination_condition
        my_sol = 'null'
        runtime = results.solver.time        
    return OptimalPrediction, my_sol, runtime


        
#%%

def lr_svm_baron_rho_new(svm, lr_info, rho_new, true_function, X_train, min_max_scaler, timelimit = 1800, objective = minimize, starting_point = None):
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    SVM constraints
    """  
    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']

    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma

    # Create a new model
    model = ConcreteModel("PMO(LR)-SVM_baron w/rho_new")
    
    # Create variables
    model.features_dim = range(len(LB))
    model.x = Var(model.features_dim, bounds = (0.0, 1.0))

    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))
    
    if starting_point != None:
        for i in range(len(LB)):
            model.x[i] = starting_point['x'][i]
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]

    # SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho)        
   
    # Set objective
    model.obj = Objective(expr = intercept + sum(coef[i] * model.x[i] for i in model.features_dim), sense = minimize)
    # Optimize model
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")

    my_sol = []
    for i in model.x: 
        my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = intercept + sum(coef[i] * model.x[i].value for i in model.features_dim) 
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = 'null'
        TrueValue_OptimalPrediction = results.solver.termination_condition
        runtime = results.solver.time           
    return OptimalPrediction, TrueValue_OptimalPrediction, runtime 
    
#%%

def lr_svm_baron_rho_new_real_data(svm, lr_info, rho_new, X_train, min_max_scaler, timelimit = 1800, objective = maximize, starting_point = None):
    """
    OPPM (Linear Regression), lr_info is a dictionary storing information of a pre-trained linear regression model
    SVM constraints
    utilized for a real-world dataset
    """  
    
    LB = lr_info['LB']
    UB = lr_info['UB']
    intercept = lr_info['intercept']
    coef = lr_info['coef']

    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma

    # Create a new model
    model = ConcreteModel("PMO(LR)-SVM_baron w/rho_new")
    
    # Create variables
    model.features_dim = range(len(LB))
    model.x = Var(model.features_dim, bounds = (0.0, 1.0))

    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))
    
    if starting_point != None:
        for i in range(len(LB)):
            model.x[i] = starting_point['x'][i]
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]

    # SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho)   
         
   
    # Set objective
    model.obj = Objective(expr = intercept + sum(coef[i] * model.x[i] for i in model.features_dim), sense = minimize)
    # Optimize model
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")

    
    my_sol = []
    for i in model.x: 
        my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = intercept + sum(coef[i] * model.x[i].value for i in model.features_dim) 
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = results.solver.termination_condition
        my_sol = 'null'
        runtime = results.solver.time           
    return OptimalPrediction, my_sol, runtime 
    

#%%

def rf_svm_baron_rho_new(svm, rf_info, rho_new, true_function, X_train, min_max_scaler, objective = minimize, timelimit = 1800, starting_point = None): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    SVM constraints
    """   

    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
        
    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma
    model = ConcreteModel("PMO(RF)-SVM_baron w/rho_new")

    # Create variables
    
    model.features_dim = range(len(LB))
    model.x = Var(model.features_dim, bounds = (0.0, 1.0))
    model.z_LB = Var(model.features_dim, bounds = (0.0, 1.0))
    model.z_UB = Var(model.features_dim, bounds = (0.0, 1.0))
    
    ### tree relevant variables: if leafnode[i][j] is selected
    model.leafnodes_dim = Set(initialize=list((t, l) for t in range(len(trees)) for l in range(len(leafnodes[t]))))
    model.leafnode = Var(model.leafnodes_dim, within = Binary)

    # svm relevant variables
    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))

    if starting_point != None:
        for i in range(len(LB)):
            model.x[i] = starting_point['x'][i]
            model.z_LB[i] = starting_point['z_LB'][i]    
            model.z_UB[i] = starting_point['z_UB'][i]
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                model.leafnode[t,l] = starting_point['leafnode'][t][l]  
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]
            
    # Create constraints
    model.oneleafpertree = ConstraintList()
    for t in range(len(trees)):
        model.oneleafpertree.add(expr = sum(model.leafnode[t,l] for l in range(len(leafnodes[t]))) == 1)
        
    model.boundeachfeaturepertree = ConstraintList()
    for f in range(len(LB)):
        for t in range(len(trees)):            
            model.boundeachfeaturepertree.add(expr = sum(leafnodes[t][l].lb[f] * model.leafnode[t,l] for l in range(len(leafnodes[t]))) <= model.z_LB[f])
            model.boundeachfeaturepertree.add(expr = 1 - sum((1 - leafnodes[t][l].ub[f]) * model.leafnode[t,l] for l in range(len(leafnodes[t]))) >= model.z_UB[f])

    model.boundeachfeature = ConstraintList()
    for f in range(len(LB)):
        model.boundeachfeature.add(expr = model.z_LB[f] <= model.z_UB[f] - 1e-4)
    
    model.boundfeatureinregion = ConstraintList()
    for f in range(len(LB)):
        model.boundfeatureinregion.add(expr = model.x[f] * 2 == model.z_LB[f] + model.z_UB[f])

    ### SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho)    
        
    # Set objective
    model.obj = Objective(expr = sum((model.z_LB[f] - model.z_UB[f]) for f in range(len(LB))) + sum(leafnodes[t][l].value * model.leafnode[t,l] for t in range(len(trees)) for l in range(len(leafnodes[t]))), sense = minimize)
    
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")

    my_sol = []
    for i in model.x: 
        my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = sum(leafnodes[t][l].value * model.leafnode[t,l].value for t in range(len(trees)) for l in range(len(leafnodes[t]))) / len(trees)
        print('optimal obj in scale of [0, 1]: ', OptimalPrediction)
        TrueValue_OptimalPrediction = true_function(reverse_minmaxed_sol)
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = 'null'
        TrueValue_OptimalPrediction = results.solver.termination_condition
        runtime = results.solver.time
    return OptimalPrediction, TrueValue_OptimalPrediction, runtime


#%%

def rf_svm_baron_rho_new_real_data(svm, rf_info, rho_new, X_train, min_max_scaler, objective = maximize, timelimit = 1800, starting_point = None): 
    """
    OPPM (Random Forest), rf_info is a dictionary storing information of a pre-trained random forest model
    SVM constraints
    utilized for a real-world dataset
    """   
    
    LB = rf_info['LB']
    UB = rf_info['UB']
    trees = rf_info['trees']
    leafnodes = rf_info['leafnodes']
    splitnodes = rf_info['splitnodes']
        
    rho = rho_new
    sv = svm.support_vectors_.tolist()
    alpha = svm.dual_coef_.ravel().tolist()
    gamma = svm._gamma
    model = ConcreteModel("PMO(RF)-SVM_baron w/rho_new")

    # Create variables
    
    model.features_dim = range(len(LB))
    model.x = Var(model.features_dim, bounds = (0.0, 1.0))
    model.z_LB = Var(model.features_dim, bounds = (0.0, 1.0))
    model.z_UB = Var(model.features_dim, bounds = (0.0, 1.0))
    
    ### tree relevant variables: if leafnode[i][j] is selected
    model.leafnodes_dim = Set(initialize=list((t, l) for t in range(len(trees)) for l in range(len(leafnodes[t]))))
    model.leafnode = Var(model.leafnodes_dim, within = Binary)

    # svm relevant variables
    model.sv_dim = range(len(sv))
    model.k = Var(model.sv_dim, bounds = (exp(-gamma * len(LB)), 1.0))    
    model.d = Var(model.sv_dim, bounds = (0.0, len(LB)))

    if starting_point != None:
        for i in range(len(LB)):
            model.x[i] = starting_point['x'][i]
            model.z_LB[i] = starting_point['z_LB'][i]    
            model.z_UB[i] = starting_point['z_UB'][i]
        for t in range(len(trees)):
            for l in range(len(leafnodes[t])):
                model.leafnode[t,l] = starting_point['leafnode'][t][l]  
        for i in range(len(sv)):
            model.k[i] = starting_point['k'][i]
            model.d[i] = starting_point['d'][i]
            
    # Create constraints
    model.oneleafpertree = ConstraintList()
    for t in range(len(trees)):
        model.oneleafpertree.add(expr = sum(model.leafnode[t,l] for l in range(len(leafnodes[t]))) == 1)
        
    model.boundeachfeaturepertree = ConstraintList()
    for f in range(len(LB)):
        for t in range(len(trees)):            
            model.boundeachfeaturepertree.add(expr = sum(leafnodes[t][l].lb[f] * model.leafnode[t,l] for l in range(len(leafnodes[t]))) <= model.z_LB[f])
            model.boundeachfeaturepertree.add(expr = 1 - sum((1 - leafnodes[t][l].ub[f]) * model.leafnode[t,l] for l in range(len(leafnodes[t]))) >= model.z_UB[f])

    model.boundeachfeature = ConstraintList()
    for f in range(len(LB)):
        model.boundeachfeature.add(expr = model.z_LB[f] <= model.z_UB[f] - 1e-4)
    
    model.boundfeatureinregion = ConstraintList()
    for f in range(len(LB)):
        model.boundfeatureinregion.add(expr = model.x[f] * 2 == model.z_LB[f] + model.z_UB[f])

    ### SVM constraints
    model.svm_distance = ConstraintList()
    for i in range(len(sv)):
        model.svm_distance.add(expr = sum((sv[i][j] - model.x[j]) * (sv[i][j] - model.x[j]) for j in model.features_dim) == model.d[i])      
 
    model.svm_kernal = ConstraintList()
    for i in range(len(sv)):
        model.svm_kernal.add(expr = exp(-gamma * model.d[i]) == model.k[i])
        
    model.svm_decision_function = Constraint(expr = sum(alpha[i] * model.k[i] for i in model.sv_dim) >= rho)    
        
    # Set objective
    model.obj = Objective(expr = sum((model.z_LB[f] - model.z_UB[f]) for f in range(len(LB))) + sum(leafnodes[t][l].value * model.leafnode[t,l] for t in range(len(trees)) for l in range(len(leafnodes[t]))), sense = minimize)
    
    opt = SolverFactory('baron', executable="C:/baron/baron.exe")
    opt.options["threads"] = 1
    opt.options["maxtime"] = timelimit
    opt.options["epsa"] = 1e-4
    results = opt.solve(model, tee=True, logfile = "baron_result.txt")

    my_sol = []
    for i in model.x: 
        my_sol.append(model.x[i].value)

    if my_sol[0] != None:
        reverse_minmaxed_sol = (min_max_scaler.inverse_transform)([my_sol])[0]
        OptimalPrediction = sum(leafnodes[t][l].value * model.leafnode[t,l].value for t in range(len(trees)) for l in range(len(leafnodes[t]))) / len(trees)
        print('optimal obj in scale of [0, 1]: ', OptimalPrediction)
        runtime = results.solver.time
    else:
        print('Solver Status: ',  results.solver.status)
        print('Termination Condition: ',  results.solver.termination_condition)
        OptimalPrediction = results.solver.termination_condition
        my_sol = 'null'
        runtime = results.solver.time
    return OptimalPrediction, my_sol, runtime
