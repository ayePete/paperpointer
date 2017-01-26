## model for Python version of optimal journal submission pathways
## Vivek Srikrishnan (srikrish@psu.edu)

import numpy as np
import os
import pandas as pd
import xarray as xr
from itertools import permutations
import sys
import timeit
import random

random.seed(0)

# show progress bar (from Stack Overflow's Vladimir Ignatyev)
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

# compares expected citation score of submitting to journal 1 before journal 2
def score(accept_probs,     # array of acceptance probabilities for each journal
          accept_times,     # array of expected times until publication for each journal
          impact_factors,   # array of impact factors for each journal
          T,                # time horizon over which we care about citations
          tR,               # time for each revision
          s):               # scooping probability
    
    impact_factors = impact_factors/365 # convert IFs to a measure of expected citations per day
    score = accept_probs[0]*impact_factors[0]*(T-accept_times[0] + (1-accept_probs[0])*np.power(1-s,tR+accept_times[0])*accept_times[1]*impact_factors[1]*(T-accept_times[0]-tR-accept_times[1]))
    return score

# computes ranking of expected citation scores across all journals. Doesn't require multiple calls to score().
def rank_journals(accept_probs,
                  accept_times,
                  impact_factors,
                  T,
                  tR,
                  s):
    
    impact_factors = impact_factors/365 # convert IFs to a measure of expected citations per day
    scores = (accept_probs*impact_factors*(1-accept_times/T))/(1-(1-accept_times/T-tR/T)*(1-accept_probs)*np.power(1-s,tR+accept_times))
    rankings = np.argsort(scores)[::-1]
    return rankings

# compute expected citations over time horizon for a journal submission pathway
def citations(accept_probs,     # array of acceptance probabilities for each journal in pathway
              accept_times,     # array of expected times until publication for each journal in pathway
              impact_factors,   # array of impact factors for each journal in pathway
              T,                # time horizon over which we care about citations
              tR,               # time for each revision
              s):               # scooping probability
    
    impact_factors = impact_factors/365 # convert IFs to a measure of expected citations per day
    submit_prob = np.insert((1-accept_probs[:-1])*np.power(1-s,accept_times[:-1]+tR),0,1)
    cum_accept_prob = accept_probs*np.cumprod(submit_prob)
    time_before_decision = accept_times+tR
    time_before_decision[0] -= tR
    remaining_citation_time = np.maximum(T-np.cumsum(time_before_decision),0)
    if np.sum(cum_accept_prob) >= 1:
        expected_citations = np.sum(impact_factors*remaining_citation_time*cum_accept_prob)/np.sum(cum_accept_prob)
    else:
        expected_citations = np.sum(impact_factors*remaining_citation_time*cum_accept_prob)
        
    return expected_citations

# compute expected number of submissions over time horizon for a journal submission pathway
def submissions(accept_probs,   # array of acceptance probabilities for each journal in pathway
                accept_times,   # array of expected times until publication for each journal in pathway
                T,              # time horizon over which we care about citations
                tR,             # time for each revision
                s):             # scooping probability
    
    submit_prob = np.insert((1-accept_probs[:-1])*np.power(1-s,accept_times[:-1]+tR),0,1)
    cum_accept_prob = accept_probs*np.cumprod(submit_prob)
    time_before_decision = accept_times+tR
    time_before_decision[0]-=tR
    within_horizon = 0.5*(1+np.sign(T-np.cumsum(time_before_decision)))
    within_horizon = within_horizon.astype(float)
    within_horizon = np.floor(within_horizon)
    num_submissions = np.arange(1,np.size(accept_probs)+1)
    expected_submissions = np.sum(num_submissions*cum_accept_prob*within_horizon)/np.sum(cum_accept_prob*within_horizon)
    return expected_submissions

# compute expected time until paper is accepted for a journal submission pathway
def tot_accept_time(accept_probs,
                    accept_times,
                    T,
                    tR,
                    s):
    
    submit_prob = np.insert((1-accept_probs[:-1])*np.power(1-s,accept_times[:-1]+tR),0,1)
    cum_accept_prob = accept_probs*np.cumprod(submit_prob)
    time_before_decision = accept_times+tR
    time_before_decision[0]-=tR
    within_horizon = 0.5*(1+np.sign(T-np.cumsum(time_before_decision)))
    within_horizon = within_horizon.astype(float)
    within_horizon = np.floor(within_horizon)
    expected_time = np.sum(np.cumsum(time_before_decision)*cum_accept_prob*within_horizon)/np.sum(cum_accept_prob*within_horizon)
    return expected_time

# Model 2: multi-objective analysis of the three objectives

# generate journal submission sequence chains. the transition probabilities are given by the product of the ratios
# of all three objectives. 
def generate_chains(T,          # time horizon over which we care about citations
                  tR,           # time for each revision
                  s,            # scooping probability
                  data,         # journal data as a pandas dataframe; columns should be Journal Name, Accept Rate, Submission Time (days), impact factor
                  n_chains,     # number of chains to generate
                  chain_len,    # length of each chain
                  n_iter):      # number of proposals for each chain

    n_journals = data.shape[0]
    # sample first two journals for each sequence
    chains = pd.DataFrame(index=np.arange(0,n_chains),columns=np.arange(0,chain_len))
    samples = list(permutations(np.arange(0,n_journals),2))
    chains.ix[:,:1] = [[sample[0],sample[1]] for sample in samples]
    # fill in rest of initial chains
    j_inds = np.arange(0,n_journals)
    chains.ix[:,2:] = [np.random.choice(j_inds[~np.in1d(j_inds,chains.loc[chain,:1])],chain_len-2,replace=False) for chain in range(n_chains)]
    # for each chain, propose n_iter swaps. accept with probability
    # alpha = (C_prop/C_curr)*(R_prop/R_curr)*(P_prop/P_curr),
    # so that if the proposed chain is better across all objectives, it will be accepted with probability 1
    # create data frame to hold objective values
    objs = pd.DataFrame(index=np.arange(chains.shape[0]),columns=['Citations','Submissions','Time'])
    # convert data to numpy array for use with functions
    data_array = data.as_matrix()
    for i in range(n_chains):
        prop_chains = pd.DataFrame(index=np.arange(n_chains+i*n_iter,n_chains+(i+1)*n_iter),columns=np.arange(chain_len))
        prop_objs = pd.DataFrame(index=np.arange(n_chains+i*n_iter,n_chains+(i+1)*n_iter),columns=['Citations','Submissions','Time'])
        curr_chain = chains.ix[i,:]
        # compute objectives for current chain
        C_curr = citations(data_array[curr_chain,1],data_array[curr_chain,2],data_array[curr_chain,3],T,tR,s)
        R_curr =  submissions(data_array[curr_chain,1],data_array[curr_chain,2],T,tR,s)
        P_curr =  tot_accept_time(data_array[curr_chain,1],data_array[curr_chain,2],T,tR,s)
        # store objectives
        objs.ix[i,'Citations'] = C_curr
        objs.ix[i,'Submissions'] = R_curr
        objs.ix[i,'Time'] = P_curr
        # generate new chain proposals
        for j in range(n_iter):
            swap_out_ind = np.random.choice(chain_len,1) # sample chain indices to swap in this iteration
            swap_in_journal = np.random.choice(np.delete(j_inds,curr_chain[swap_out_ind]),1)[0]
            prop_chain = curr_chain
            if np.in1d(swap_in_journal,curr_chain)[0]:
                swap_in_ind = np.argwhere(curr_chain == swap_in_journal)[0]
                prop_chain[swap_out_ind],prop_chain[swap_in_ind] = prop_chain[swap_in_ind],prop_chain[swap_out_ind] # swap sampled elements
            else:
                prop_chain[swap_out_ind] = swap_in_journal
            prop_chains.ix[j+n_chains+i*n_iter,:] = prop_chain # store proposed chain
            # compute objectives for proposed chain
            C_prop = citations(data_array[prop_chain,1],data_array[prop_chain,2],data_array[prop_chain,3],T,tR,s)
            R_prop =  submissions(data_array[prop_chain,1],data_array[prop_chain,2],T,tR,s)
            P_prop =  tot_accept_time(data_array[prop_chain,1],data_array[prop_chain,2],T,tR,s)
            # store objectives for proposed chains
            prop_objs.ix[j+n_chains+i*n_iter,'Citations'] = C_prop
            prop_objs.ix[j+n_chains+i*n_iter,'Submissions'] = R_prop
            prop_objs.ix[j+n_chains+i*n_iter,'Time'] = P_prop
            # calculate acceptance probability to replace "best" chain with proposed chain
            p_accept = (C_prop/C_curr)*(R_prop/R_curr)*(P_curr/P_prop)
            draw = np.random.uniform(0,1)
            # if proposal accepted, replace current chain with it
            if draw < p_accept:
                curr_chain = prop_chain
                C_curr = C_prop
                R_curr = R_prop
                P_curr = P_prop
        # Set chain and objectives to best results
        chains = pd.concat([chains,prop_chains],axis=0,ignore_index=True)
        objs = pd.concat([objs,prop_objs],axis=0,ignore_index=True) 
        # update progress bar
        progress(i,n_chains,suffix='Complete')
    # merge chains and objectives for return    
    df = pd.concat([chains,objs],axis=1)
    return df
    
T0 = int(sys.argv[1])
# start timer
start = timeit.default_timer()

# read data from excel file. change path to appropriate paperpointer path for file system
#os.chdir('d:\\research\\paperpointer')  # change working directory to main paperpointer directory
xls = pd.ExcelFile('data/Salinas_Munch_2015_S1_Table.xlsx') # open excel data file using pandas
data = xls.parse(xls.sheet_names[0])  # read in data as pandas dataframe

n_journals = data.shape[0]
n_chains = n_journals*(n_journals-1)    # generate a chain for each combination of first two journals
n_iter = 10000  # number of proposals for each chain
chain_len = 5

# set time horizon (T) in days, revision time (tR) in days, scooping probability (s)
T = T0*365
tR = 30
s = 0.001

# generate submission chains
df = generate_chains(T,tR,s,data,n_chains,chain_len,n_iter)
# write data frame to netcdf
df_xr = xr.Dataset.from_dataframe(df)
# Turn variable names into strings for netCDF writing
print('\r Saving to netCDF...')
var_names = {}
new_var_names = ["j%d" % number for number in np.arange(chain_len)]
new_var_names.extend(['Citations','Submissions','Review Time'])
old_var_names = [number for number in np.arange(chain_len)]
old_var_names.extend(['Citations', 'Submissions', 'Time'])
all_var_names = zip(old_var_names,new_var_names)
for old, new in all_var_names:
    var_names[old] = new

df_xr.rename(var_names,inplace=True)

df_xr.to_netcdf('chains-'+str(T0)+'.nc',mode='w')

# end timer and print runtime
stop = timeit.default_timer()
print("Runtime: " + str(stop-start) + " seconds")
