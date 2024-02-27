"""
运行前请移除results文件夹中的所有数据结果
"""

import os
import Simulator_optlang
import pandas as pd
import glob
import time

from iBridgeFunctions import *
from cobra.io import read_sbml_model

import re

filename = r"C:\Users\Victor\PycharmProjects\pythonProject\iBridge\input\ecYeastGEM_batch.xml"
#filename=r"C:\Users\Victor\PycharmProjects\pythonProject\iBridge\input\iJO1366.xml"
biomass_reaction = 'r_2111'
target_reaction = 'r_2024'
#biomass_reaction='BIOMASS_Ec_iJO1366_core_53p95M'
#target_reaction='EX_ptrc_e'

input_dir = './input'
output_dir = './results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

constrict={}

simulator = Simulator_optlang.Simulator()
model_metabolites, model_reactions, Smatrix, lower_boundary_constraints, upper_boundary_constraints, objective_reaction = simulator.read_model(filename)

corr_output = '%s/corr_%s.csv'%(output_dir,target_reaction)
cov_output = '%s/cov_%s.csv'%(output_dir, target_reaction)

flux_distribution_dict = generate_multiple_flux_profiles(simulator, biomass_reaction, target_reaction, constrict)

flux_df = pd.DataFrame.from_dict(flux_distribution_dict)
flux_corr_df = flux_df.abs().T.corr()
flux_cov_df = flux_df.abs().T.cov()

flux_corr_df = flux_corr_df[target_reaction].loc[simulator.model_reactions]
flux_corr_df.to_csv(corr_output,header=0)
flux_cov_df = flux_cov_df[target_reaction].loc[simulator.model_reactions]
flux_cov_df.to_csv(cov_output,header=0)

write_header(corr_output, target_reaction)
write_header(cov_output, target_reaction)

df = pd.read_csv(cov_output, index_col=0)

#ecModel = read_sbml_model(r"C:\Users\Victor\PycharmProjects\pythonProject\iBridge\input\iJO1366.xml")
ecModel = read_sbml_model(r"C:\Users\Victor\PycharmProjects\pythonProject\iBridge\input\ecYeastGEM_batch.xml")
met_info = metabolite_set(ecModel)

final_dic = {}
fluxsum_dic = calculate_MetScore_sum(ecModel,  dict(df[target_reaction]))
final_dic[target_reaction]=fluxsum_dic

metscore_df = pd.DataFrame.from_dict(final_dic)
metscore_df.to_csv('%s/MetScore_%s.csv'%(output_dir, target_reaction))

flux_corr_df = pd.read_csv(corr_output, index_col=0)
flux_cov_df = pd.read_csv(cov_output, index_col=0)

output_file = '%s/Final_MetScore_%s.txt'%(output_dir, target_reaction)

select_candidates(ecModel, target_reaction, output_file, metscore_df, flux_corr_df, flux_cov_df)

df2 = pd.read_table(os.path.abspath(output_file))

files = glob.glob('%s/Final_*.txt'%output_dir)
for filename in files:
    basename = os.path.basename(filename)
    df2 = pd.read_table(filename)
    make_candidate_reaction_sets(df2, basename, input_dir, output_dir)

files = glob.glob('%s/application_results/*.txt' % output_dir)

for each_file in files:
    s = time.time()
    basename = os.path.basename(each_file)
    target_reaction_id = basename.split('Final_MetScore_')[1].strip().replace('.txt', '')

    corr_file = glob.glob('%s/corr*%s*.csv' % (output_dir, target_reaction_id))[0]
    cov_file = glob.glob('%s/cov*%s*.csv' % (output_dir, target_reaction_id))[0]

    flux_corr_df = pd.read_csv(corr_file, index_col=0)
    flux_cov_df = pd.read_csv(cov_file, index_col=0)

    print('Target reaction: %s' % (target_reaction_id))

    up_reaction_list = check_up_reaction(each_file, met_info, flux_corr_df, flux_cov_df, output_dir)
    down_reaction_list = check_down_reaction(each_file, met_info, flux_corr_df, flux_cov_df, output_dir)

    e = time.time()
    print('Elapsed time: %fs' % (e - s))

up_reactions_matched = []
down_reactions_matched = []

for up_reaction in up_reaction_list:
    if re.match(r'r_\d{4}$', up_reaction):
        up_reactions_matched.append(up_reaction)

for down_reaction in down_reaction_list:
    if re.match(r'r_\d{4}$', down_reaction):
        down_reactions_matched.append(down_reaction)

up_reactions = pd.DataFrame(data=up_reactions_matched, columns=['Up'])
down_reactions = pd.DataFrame(data=down_reactions_matched, columns=['Down'])

up_reactions.to_csv('./results/application_result2/up_reactions.csv', index=0)
down_reactions.to_csv('./results/application_result2/down_reactions.csv', index=0)