import os
import numpy as np 
import pandas as pd
import mlrose_hiive as mlrose
import time
import matplotlib.pyplot as plt

RHC=[]
GA=[]
SA=[]
MIMIC=[]
size_range = range(20,100,20)

for size in size_range:
    fitness = mlrose.OneMax()
    problem_fit = mlrose.DiscreteOpt(length = size,fitness_fn = fitness,maximize = True,max_val = 2)
    
    st_time=time.time()
    RHC_curve = mlrose.random_hill_climb(problem_fit,restarts=10*size,max_attempts=10,max_iters=size*10,init_state=None,curve=True)
    et_time=time.time()
    RHC.append([size,et_time-st_time,RHC_curve[2][-1,0]])
    
    st_time=time.time()
    GA_curve = mlrose.genetic_alg(problem_fit,pop_size=10*size,mutation_prob=0.4,max_attempts=10,max_iters=size*10,curve=True)
    et_time=time.time()
    GA.append([size,et_time-st_time,GA_curve[2][-1,0]])
    
    st_time=time.time()
    SA_curve = mlrose.simulated_annealing(problem_fit,schedule=mlrose.GeomDecay(),max_attempts=10,init_state=None,max_iters=size*10,curve=True)
    et_time=time.time()
    SA.append([size,et_time-st_time,SA_curve[2][-1,0]])
    
    st_time=time.time()
    MIMIC_curve = mlrose.mimic(problem_fit,pop_size=10*size, keep_pct=0.2,max_attempts=10,max_iters=size*10,curve=True)
    et_time=time.time()
    MIMIC.append([size,et_time-st_time,MIMIC_curve[2][-1,0]])
RHC_df = pd.DataFrame(RHC)[1]
GA_df = pd.DataFrame(GA)[1]
SA_df = pd.DataFrame(SA)[1]
MIMIC_df = pd.DataFrame(MIMIC)[1]
df = pd.concat([RHC_df,GA_df,SA_df,MIMIC_df], ignore_index=True, axis=1)
df.index=size_range
df.columns = ["RHC","GA","SA","MIMIC"]
df.plot(marker='o',xlabel="Problem Size",ylabel="Time",title="Time Taken vs Problem Size",figsize=(12,6))
plt.savefig('onemax_time_vs_problemsize.png')
RHC_df = pd.DataFrame(RHC)[2]
GA_df = pd.DataFrame(GA)[2]
SA_df = pd.DataFrame(SA)[2]
MIMIC_df = pd.DataFrame(MIMIC)[2]
df = pd.concat([RHC_df,GA_df,SA_df,MIMIC_df], ignore_index=True, axis=1)
df.index=size_range
df.columns = ["RHC","GA","SA","MIMIC"]
df.plot(marker='o',xlabel="Problem Size",ylabel="Best Fitness value",title="Best Fitness vs Problem Size",figsize=(12,6))
plt.savefig('onemax_fitness_vs_problemsize.png')
