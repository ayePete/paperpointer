## model for Python MORDM version of optimal journal submission pathways
## Vivek Srikrishnan (srikrish@psu.edu)


import numpy as np
import os
import pandas as pd
import xarray as xr
import sys
import random
from rhodium import *
import timeit

random.seed(0)

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
    expected_citations = np.sum(impact_factors*remaining_citation_time*cum_accept_prob)/np.sum(cum_accept_prob)
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

# paperpointer model for expected citations (C), submissions (R), time under review (P)
def paperpointer(j_seq,         # sequence of journals to be evaluated,
                 T,             # time horizon over which we care about citations (days)
                 tR = 30,       # time for each revision (days)
                 s = 0.001):    # scooping probability
    
    accept_probs = np.array([j_data[s]['AcceptRate'] for s in j_seq])
    dec_time = np.array([j_data[s]['SubToDecTime_days'] for s in j_seq])
    impact_factors = np.array([j_data[s]['IF_2012'] for s in j_seq])
    exp_cite = citations(accept_probs,dec_time,impact_factors,T,tR,s)
    exp_subs = submissions(accept_probs,dec_time,T,tR,s)
    exp_review_time = tot_accept_time(accept_probs,dec_time,T,tR,s)
    
    return (exp_cite,exp_subs,exp_review_time)

# load journal data
# read data from excel file. change path to appropriate paperpointer path for file system
os.chdir('d:\\research\\paperpointer')  # change working directory to main paperpointer directory

T0 = int(sys.argv[1])

start = timeit.default_timer()

xls = pd.ExcelFile('data/Salinas_Munch_2015_S1_Table.xlsx') # open excel data file using pandas
data_pd = xls.parse(xls.sheet_names[0])  # read in data as pandas dataframe
j_data = data_pd.set_index('Journal').to_dict(orient='index')

# define model for rhodium
model = Model(paperpointer)

model.parameters = [Parameter("j_seq"),
                    Parameter("T", default_value = T0*365),
                    Parameter("tR"),
                    Parameter("s")]

model.responses = [Response("exp_cite",Response.MAXIMIZE),
                   Response("exp_subs",Response.MINIMIZE),
                   Response("exp_review_time",Response.MINIMIZE)]

# specify model Lever
model.levers = [SubsetLever("j_seq",options=j_data.keys(),size=5)]

# optimize using NSGAII
output = optimize(model,"NSGAII",1000000)
print("Found " + str(len(output)) + " optimal policies!")
print(output)

# write Pareto optimal solutions to csv
output_df = output.as_dataframe()
output_xr = xr.Dataset.from_dataframe(output_df)
output_xr.to_netcdf('chains-Rhodium-'+str(T0)+'.nc',mode='w')    

fig = scatter2d(model, output)
plt.show()

fig = parallel_coordinates(model, output, colormap="rainbow", target="top")
plt.show()

stop = timeit.default_timer()
print("Runtime: " + str(stop-start) + " seconds")