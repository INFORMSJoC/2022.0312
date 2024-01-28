[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Constraint Learning to Define Trust Regions in Optimization Over Pre-Trained Predictive Models
This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The purpose of this repository is to share the instances and results used in the paper "Constraint Learning to Define Trust Regions in Optimization Over Pre-Trained Predictive Models", authored by Chenbo Shi, Mohsen Emadikhiav, Leonardo Lozano, and David Bergman.

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
  year =          {2023},
  doi =           {10.1287/ijoc.2022.0312.cd},
  note =          {Available for download at https://github.com/INFORMSJoC/2022.0312},
}  
```

## Dataset 
The folder of [data](data/) contains all the datasets used in the paper, including the wine dataset used in the real world application ([winequality-red.csv](data/winequality-red.csv)) and all the 70 training datasets for seven benchmark functions (under the subfolder of [data/synthetic_data](data/synthetic_data)). 

For each function, we generate 10 different datasets consisting of 1000 points, each generated with a different randomly drawn covariance matrix by specifying the random state. For each dataset, we scale the independent variables to [0, 1], standardize the dependent variable, and randomly split the data into a training set and a test set in the ratio of 7 : 3. Each .xlsx file in [data/synthetic_data](data/synthetic_data) corresponds to the part of the training set, containing 700 points. For each point, column $0$ to column $n-1$ (both $0$ and $n-1$ are column names) records the values of the scaled independent variables, and column $y$ records the value of the standardized dependent variable. 

## Results 
The folder of [results](results/) contains the solutions ([solutions_wine.csv](results/solutions_wine.csv)) and colored figures from the real world application and the experiment results using the synthetic data ([results_experiment.xlsx](results/results_experiment.xlsx)). 

For experiment results on synthetic data, each record in the "AllInstanceResults" spreadsheet of [results_experiment.xlsx](results/results_experiment.xlsx) records the optimization results obtained from solving a problem instance, which is defined by a predictive model (Column C) fitted to a dataset, with a model configuration (Column D) at a specific tightness level (Column E). 

The results include predicted outcome $F(x^\ast_C)$ in Column N, true outcome $f(x^\ast_C)$ in Column O, where $x^\ast_C$ is the solution found by the OPPM model configuration selected, running time in Column R, the percent improvement over **$BASE$** in column Q, and the gap in Column S.

For results on wine data, each figure in [results](results/) folder represents a predictive model fitted to the wine data and displays training input data, and solutions obtained from the **$BASE$** model and two model configurations with **$IF$** and **$MD$** constraints (set at tightness level 6).



