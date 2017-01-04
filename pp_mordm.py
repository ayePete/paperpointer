import numpy as np
import os
import pandas as pd
import xarray as xr
import sys
import random
from rhodium import *

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
def paperpointer(r,             # vector of weights for each element of journal data
                 T = 2*365,     # time horizon over which we care about citations (days)
                 tR = 30,       # time for each revision (days)
                 s = 0.001):    # scooping probability
    
    xls = pd.ExcelFile('data/Salinas_Munch_2015_S1_Table.xlsx') # open excel data file using pandas
    data = xls.parse(xls.sheet_names[0])  # read in data as pandas dataframe
    data_np = data.as_matrix()
    r_np = np.asarray(r)
    score = np.dot(data_np[:,1:4],r_np.T) # compute journal score as linear combination of journal data
    j_seq = np.argsort(score)[::-1]   # sort by score in decreasing order
    exp_cite = citations(data_np[j_seq,1],data_np[j_seq,2],data_np[j_seq,3],T,tR,s)
    exp_subs = submissions(data_np[j_seq,1],data_np[j_seq,2],T,tR,s)
    exp_review_time = tot_accept_time(data_np[j_seq,1],data_np[j_seq,2],T,tR,s)
    
    return (exp_cite,exp_subs,exp_review_time)

# load journal data
# read data from excel file. change path to appropriate paperpointer path for file system
os.chdir('d:\\research\\paperpointer')  # change working directory to main paperpointer directory

# define model for rhodium
model = Model(paperpointer)

model.parameters = [Parameter("r"),
                    Parameter("T"),
                    Parameter("tR"),
                    Parameter("s")]

model.responses = [Response("exp_cite",Response.MAXIMIZE),
                   Response("exp_subs",Response.MINIMIZE),
                   Response("exp_review_time",Response.MINIMIZE)]

model.constraints = []

# specify r as model Lever
model.levers = [RealLever("r",min_value=-1,max_value=1,length=3)]

# optimize using NSGAII
output = optimize(model,"NSGAII",100000)
print("Found " + str(len(output)) + " optimal policies!")

# write Pareto optimal solutions to csv
output_df = output.as_dataframe()
output_df.to_pickle('policies-T2')    

fig = scatter2d(model, output)
plt.show()

fig = scatter3d(model, output)
plt.show()
fig = parallel_coordinates(model, output, colormap="rainbow", target="top")
plt.show()