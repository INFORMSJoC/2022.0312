[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Constraint Learning to Define Trust Regions in Optimization Over Pre-Trained Predictive Models
This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The purpose of this repository is to share the codes, instances, and results used in the paper "Constraint Learning to Define Trust Regions in Optimization Over Pre-Trained Predictive Models", authored by Chenbo Shi, Mohsen Emadikhiav, Leonardo Lozano, and David Bergman.

## Cite
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.
[https://doi.org/10.1287/ijoc.2022.0312](https://doi.org/10.1287/ijoc.2022.0312)

[https://doi.org/10.1287/ijoc.2022.0312.cd](https://doi.org/10.1287/ijoc.2022.0312.cd)

Below is the BibTex for citing this snapshot of the repository.
```
@article{contrslearning_oppm_data,
  author =        {Shi, Chenbo and Emadikhiav, Mohsen and Lozano, Leonardo and Bergman, David},
  publisher =     {INFORMS Journal on Computing},
  title =         {Constraint Learning to Define Trust Regions in  Optimization Over Pre-Trained Predictive Models},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0312.cd},
  note =          {Available for download at https://github.com/INFORMSJoC/2022.0312},
}  
```

## Dataset 
The folder of [data](data/) contains all the datasets used in the paper, including the wine dataset used in the real world application ("data/winequality-red.csv") and all the 70 training datasets for seven benchmark functions (under the subfolder of "data/synthetic_data"). 

For each function, we generate 10 different datasets consisting of 1000 points, each generated with a different randomly drawn covariance matrix by specifying the random state. For each dataset, we scale the independent variables to [0, 1], standardize the dependent variable, and randomly split the data into a training set and a test set in the ratio of 7 : 3. Each .xlsx file in "data/synthetic_data" corresponds to the part of the training set, containing 700 points. For each point, column $0$ to column $n-1$ (both $0$ and $n-1$ are column names) records the values of the scaled independent variables, and column $y$ records the value of the standardized dependent variable. 

## Results 
The folder of [results](results/) contains the solutions ("results/solutions_wine.csv") and colored figures from the real world application and the experiment results using the synthetic data ("results/results_experiment.xlsx"). 

For experiment results on synthetic data, each record in the "AllInstanceResults" spreadsheet of "results/results_experiment.xlsx" records the optimization results obtained from solving a problem instance, which is defined by a predictive model (Column C) fitted to a dataset, with a model configuration (Column D) at a specific tightness level (Column E). 

The results include predicted outcome $F(x^\ast_C)$ in Column N, true outcome $f(x^\ast_C)$ in Column O, where $x^\ast_C$ is the solution found by the OPPM model configuration selected, running time in Column R, the percent improvement over **$BASE$** in column Q, and the gap in Column S.

For results on wine data, each figure in "results" folder represents a predictive model fitted to the wine data and displays training input data, and solutions obtained from the **$BASE$** model and two model configurations with **$IF$** and **$MD$** constraints (set at tightness level 6).

## Replicating
The folder of [src](src/) contains all the code that generates synthetic data and implements the optimization models based on three predictive models (linear regression, random forest, and neural network) with different types of trust-region constraints.  

To run the code and fully replicate the experiments, you will need to make sure that you have valid licenses of <code>Gurobi</code> and <code>BARON</code>, and the following dependencies installed:\
<code>Python 3.7.3</code> <code>scikit-learn</code> <code>numpy</code> <code>scipy</code> <code>pandas</code> <code>random</code> <code>gurobipy</code> <code>pyomo</code>

[main_synthetic_data.py](src/main_synthetic_data.py) corresponds to the experiment described in Section 4.1 - 4.6, while [main_real_data.py](src/main_real_data.py) corresponds to Section 4.7 in the paper. The remaining files contain code used to define functions, build models, and specify trust-region constraints. Among them, we want to highlight that: 
* [predictive_modeling.py](src/predictive_modeling.py) contains code that fetches information from pre-trained predictive models, which will be fed into the OPPM models.
* [optimization_models.py](src/optimization_models.py) includes the OPPM models of **BASE** (without trust-region constraints) and variants with the following configurations: **IF**, **MD**, **KNN**, which are trust-regions proposed in this paper.
* [optimization_benchmark_models.py](src/optimization_benchmark_models.py) includes the OPPM models with the following configurations: **SVM-BC**, **CH**, **PCA**, which are trust-regions existing in literature.
* [optimization_svm_model_baron.py](src/optimization_svm_model_baron.py) includes the OPPM model with the **SVM** configuration, which is a trust-regions existing in literature and needs to be solved with <code>BARON</code>.


