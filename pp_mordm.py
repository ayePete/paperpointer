## model for Python MORDM version of optimal journal submission pathways
## Vivek Srikrishnan (srikrish@psu.edu)


import numpy as np
import os
import pandas as pd
import xarray as xr
import sys
import random
from platypus import *
from rhodium import *
import timeit

random.seed(1)

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
    
def submissions(accept_probs,   # array of acceptance probabilities for each journal in pathway
                accept_times,   # array of expected times until publication for each journal in pathway
                T,              # time horizon over which we care about citations
                tR,             # time for each revision
                s):             # scooping probability
    
    submit_prob = np.insert((1-accept_probs[:-1])*np.power(1-s,accept_times[:-1]+tR),0,1)
    cum_accept_prob = accept_probs*np.cumprod(submit_prob)
    cum_accept_prob[-1] = 1-np.sum(cum_accept_prob[:-1])
    time_before_decision = accept_times+tR
    time_before_decision[0]-=tR
    within_horizon = 0.5*(1+np.sign(T-np.cumsum(time_before_decision)))
    within_horizon = within_horizon.astype(float)
    within_horizon = np.floor(within_horizon)
    sub_max = np.nonzero(within_horizon)[0][-1]+1
    num_submissions = np.arange(1,np.size(accept_probs)+1)
    expected_submissions = np.sum(np.minimum(sub_max,num_submissions)*cum_accept_prob)
    return expected_submissions

# compute expected time until paper is accepted for a journal submission pathway
def tot_accept_time(accept_probs,
                    accept_times,
                    T,
                    tR,
                    s):
    
    submit_prob = np.insert((1-accept_probs[:-1])*np.power(1-s,accept_times[:-1]+tR),0,1)
    cum_accept_prob = accept_probs*np.cumprod(submit_prob)
    cum_accept_prob[-1] = 1-np.sum(cum_accept_prob[:-1])
    time_before_decision = accept_times+tR
    time_before_decision[0]-=tR
    expected_time = np.sum(np.minimum(T,np.cumsum(time_before_decision))*cum_accept_prob)
    return expected_time

# compute probability of not being accepted by any journal, including scooping
def rejection_probability(accept_probs,
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
    sub_max = np.nonzero(within_horizon)[0][-1]+1  
    reject_prob = 1-np.sum(cum_accept_prob[:sub_max])
    return reject_prob

# paperpointer model for expected citations (C), submissions (R), time under review (P)
def paperpointer(j_seq,         # sequence of journals to be evaluated,
                 T,             # time horizon over which we care about citations (days),
                 j_data,        # journal data as dictionary indexed by name
                 tR = 30,       # time for each revision (days)
                 s = 0.001):    # scooping probability
    
    accept_probs = np.array([j_data[jour]['AcceptRate'] for jour in j_seq])
    dec_time = np.array([j_data[jour]['SubToDecTime_days'] for jour in j_seq])
    impact_factors = np.array([j_data[jour]['IF_2012'] for jour in j_seq])
    expected_citations = citations(accept_probs,dec_time,impact_factors,T,tR,s)
    expected_submissions = submissions(accept_probs,dec_time,T,tR,s)
    expected_review_time = tot_accept_time(accept_probs,dec_time,T,tR,s)
    tot_accept_prob = 1-rejection_probability(accept_probs,dec_time,T,tR,s)
    return (expected_citations,expected_submissions,expected_review_time,tot_accept_prob)

# load journal data
# read data from excel file. change path to appropriate paperpointer path for file system
#os.chdir('d:\\research\\paperpointer')  # change working directory to main paperpointer directory

param_array = [[3,10000000],[7,10000000],[20,10000000]]

# get array index to look up T0 and NFE
array_ind = int(os.getenv('PBS_ARRAYID'))

T0 = param_array[array_ind][0]
NFE = param_array[array_ind][1]

# start timer
start = timeit.default_timer()

xls = pd.ExcelFile('data/Salinas_Munch_2015_S1_Table.xlsx') # open excel data file using pandas
data_pd = xls.parse(xls.sheet_names[0])  # read in data as pandas dataframe
j_data = data_pd.set_index('Journal').to_dict(orient='index')

# define model for rhodium
model = Model(paperpointer)

model.parameters = [Parameter("j_seq"),
                    Parameter("T", default_value = T0*365),
                    Parameter("j_data", default_value=j_data),
                    Parameter("tR"),
                    Parameter("s")]

model.responses = [Response("expected_citations",Response.MAXIMIZE),
                   Response("expected_submissions",Response.MINIMIZE),
                   Response("expected_review_time",Response.MINIMIZE),
                   Response("tot_accept_prob",Response.INFO)]

# specify model Lever
model.levers = [SubsetLever("j_seq",options=j_data.keys(),size=5)]

# optimize using NSGAII
output = optimize(model,"NSGAII",NFE)
print("Found " + str(len(output)) + " optimal policies!")
print(output)

stop = timeit.default_timer()
print("Runtime: " + str(stop-start) + " seconds")

# write Pareto optimal solutions to csv
output_df = output.as_dataframe()
output_df = pd.concat([output_df['j_seq'].apply(pd.Series),output_df.drop(['j_seq','j_data','T'],axis=1)],axis=1)
output_xr = xr.Dataset.from_dataframe(output_df)
var_names = {}
new_var_names = ["j%d" % number for number in np.arange(5)]
new_var_names.extend(['Citations','Submissions','Review Time'])
old_var_names = [number for number in np.arange(5)]
old_var_names.extend(['expected_citations', 'expected_submissions','expected_review_time'])
all_var_names = zip(old_var_names,new_var_names)
for old, new in all_var_names:
    var_names[old] = new

output_df.to_pickle('MORDM-'+str(T0)+'-'+str(NFE)+'.pkl')

output_xr.rename(var_names,inplace=True)

output_xr.to_netcdf('MORDM-'+str(T0)+'-'+str(NFE)+'.nc',mode='w')    

# get color map
# unique_j0 = list(output_df[0].unique())
# unique_count = len(unique_j0)
# hex_colors = sns.color_palette('husl',unique_count).as_hex()
# color_zip = zip(unique_j0,hex_colors)
# j_colors = {}
# for journal,color in color_zip:
#     j_colors[journal] = color

# output.apply("first_journal = j_seq[0]")
# 
# sns.set_style('dark')
# fig = scatter2d(model, output,c="first_journal",is_class=True,colors=j_colors,x='expected_citations',y='expected_submissions')
# plt.show()
# 
# fig = scatter2d(model, output,c="first_journal",is_class=True,colors=j_colors,x='expected_citations',y='expected_review_time')
# plt.show()
# 
# fig = scatter2d(model, output,c="first_journal",is_class=True,colors=j_colors,x='expected_submissions',y='expected_review_time')
# plt.show()
# 
# sns.set_style('dark')
# fig = parallel_coordinates(model, output, target="top", c="first_journal", colors=j_colors)
# plt.show()

