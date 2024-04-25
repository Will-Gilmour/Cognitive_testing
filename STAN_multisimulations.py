import matplotlib.pyplot as plt
# import pystan
import pandas as pd
import numpy as np
import os
import arviz
import pickle
import sys
import csv
import pdb

main_path = '/home/will/Documents/Apathy/STAN_Loo_v4'
scripts_path = main_path + '/Scripts/Python'
data_path_load = main_path + '/Simulation_Full'
data_path = data_path_load + '/parameters'
bayesSMEP_path = data_path + '/BayesSMEP'
simsave_path = data_path_load + '/Simulation/10'

nSubjects = 134
nTrials = 50
nSimulations = 100

# os.chdir(scripts_path)
# sim_model = pystan.StanModel(file='BayesSMEP_model_simulation.stan')

os.chdir(data_path_load)
walks = pd.read_csv('walks.csv')
post_op_walks = walks.iloc[:, 1].values # assuming the second column is the post walks
walk_to_reward_file = {
    1: 'Reward1.csv',
    2: 'Reward2.csv',
    3: 'Reward3.csv',
}
par = pd.read_csv('Parameters.csv')
beta = par['beta'].values
phi = par['phi'].values
per = par['per'].values

for subject_idx in range(nSubjects):
    subject_choices = []
    subject_rewards = []

    os.chdir(data_path_load)
    reward_file = walk_to_reward_file[post_op_walks[subject_idx]]
    re = pd.read_csv(reward_file)
    reward_curr = re.iloc[:nTrials, :].to_numpy()  # take only the first nTrials
    beta_curr = beta[subject_idx]
    phi_curr = phi[subject_idx]
    per_curr = per[subject_idx]

    my_data = {
        'nSubjects': 1,
        'nTrials': nTrials,
        'reward': reward_curr,
        'beta': beta_curr,
        'phi': phi_curr,
        'persev': per_curr
    }
    os.chdir(simsave_path)

    for sim_num in range(nSimulations):
        print(f"Currently on simulation {sim_num - 1}/{nSimulations}")
        simulated = sim_model.sampling(data=my_data, iter=1, chains=1, algorithm="Fixed_param")
        choice_sim = simulated.extract()

        choice = np.squeeze(choice_sim['choice'])
        reward_obt = np.squeeze(choice_sim['reward_obt'])

        subject_choices.append(choice)
        subject_rewards.append(reward_obt)

    Cdf = pd.DataFrame(subject_choices)
    Cdf.to_csv(f"Choice_Sim_Subject{subject_idx}.csv")
    Rdf = pd.DataFrame(subject_rewards)
    Rdf.to_csv(f"Reward_Obt_Sim_Subject{subject_idx}.csv")


