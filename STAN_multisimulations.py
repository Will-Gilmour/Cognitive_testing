import pystan
import pandas as pd
import numpy as np
import os
import sys

# Parameters for simulation
nSubjects = 1  # Please verify that the parameters.csv and walks.csv are both this length by 3 and 1 respectively!
nTrials = 300   # This will chop from the beginning of the walk to this point, max is 300
nSimulations = 100

current_dir = os.path.dirname(__file__)  # This file can be moved around to different directories and still work!
simulations_dir = os.path.join(current_dir, 'simulations')  # Will need to have the subdirectory where data gets saved!

# If having issue with the STAN file use the second line rather than the first, also need to add num_subjects back in
# Also will need to squeeze the output at the end!
# sim_model = pystan.StanModel(file='BayesSMEP_model_simulation_single.stan')
sim_model = pystan.StanModel(file='BayesSMEP_model_simulation.stan')


print("Walks being loaded in")
walks = pd.read_csv('walks.csv')
print(walks.head())
print('')

post_op_walks = walks.iloc[:, 1].values   # assuming the second column is the post walks
walk_to_reward_file = {
    1: 'Reward1.csv',
    2: 'Reward2.csv',
    3: 'Reward3.csv',
}
par = pd.read_csv('parameters.csv')
print("These are the parameters being pulled for the simulations:")
print(par.head())  # for debugging!
print('')

print("Important to note the length here should be equal to the number of subjects!")
print("Number of subjects for simulation: " + str(nSubjects))
print("Length of parameter matrix: " + str(len(par)))
print('')

beta = par['beta'].values
phi = par['phi'].values
per = par['per'].values


for subject_idx in range(nSubjects):
    subject_choices = []
    subject_rewards = []

    # os.chdir(data_path_load)
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
    print(f'Data loaded for subject {subject_idx+1}:')
    print(my_data)  # This will show the data for every single

    for sim_num in range(nSimulations):
        msg = f"Currently on simulation {sim_num + 1}/{nSimulations}"
        sys.stdout.write('\r' + msg)

        simulated = sim_model.sampling(data=my_data, iter=1, chains=1, algorithm="Fixed_param")
        choice_sim = simulated.extract()

        # Use these lines if using the 'single' version of STAN model
        choice = choice_sim['choice']
        reward_obt = choice_sim['reward_obt']

        # These lines are for the 'standard' version of the STAN model
        # choice = np.squeeze(choice_sim['choice'])
        # reward_obt = np.squeeze(choice_sim['reward_obt'])

        print(choice)
        print(reward_obt)

        # I can't run Pystan locally, so I these lines are just to get some data generated!
        # choice = np.random.randint(low=1, high=4, size=nTrials)
        # reward_obt = np.random.rand(nTrials)*100

        subject_choices.append(choice)
        subject_rewards.append(reward_obt)

    print('')
    print(f'Finished simulations for subject {subject_idx + 1}/{nSubjects}')
    print('Saving...')
    pd.DataFrame(subject_choices).to_csv(os.path.join(simulations_dir, f"Choice_Sim_Subject{subject_idx+1}.csv"))
    pd.DataFrame(subject_rewards).to_csv(os.path.join(simulations_dir, f"Reward_Obt_Sim_Subject{subject_idx+1}.csv"))
    print('')

