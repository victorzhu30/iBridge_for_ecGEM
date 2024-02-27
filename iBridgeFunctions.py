import numpy as np
from copy import deepcopy
import os
import pandas as pd


def generate_multiple_flux_profiles(simulator, biomass_rxn, target_rxn, constrict={}):
    _, _, wild_flux= simulator.run_FBA(internal_flux_minimization=True)

    _, _, flux_dic = simulator.run_FBA(new_objective=target_rxn,
                                       flux_constraints=constrict,
                                       mode='max')
    max_const = flux_dic[target_rxn]

    _, _, flux_dic = simulator.run_FBA(new_objective=target_rxn,
                                       flux_constraints=constrict,
                                       mode='min')
    min_const = flux_dic[target_rxn]

    flux_distribution_dict = {}
    count = 0
    for each_target_rxn_flux in np.linspace(min_const, max_const, 10):
        tmp_const = deepcopy(constrict)
        count += 1
        tmp_const[target_rxn] = [each_target_rxn_flux * 0.95, each_target_rxn_flux * 1.05]
        state, obj_val, opt_flux_dic = simulator.run_FBA(new_objective=biomass_rxn,
                                                        flux_constraints=tmp_const)

        if state != 2:
            print('Biomass optimization failed at count %d'%count)
            continue

        tmp_biomass_flux = opt_flux_dic[biomass_rxn]
        tmp_const[biomass_rxn] = [tmp_biomass_flux * 0.95, tmp_biomass_flux * 1.05]

        print('Count: %d\tBiomass flux: %0.6f' % (count, tmp_biomass_flux))

        state, obj_val, flux_dic = simulator.run_MOMA(wild_flux=wild_flux,
                                                    flux_constraints=tmp_const)

        if state != 2:
            print('Linear moma failed at count %d'%count)
        else:
            flux_distribution_dict[each_target_rxn_flux] = flux_dic

        if abs(flux_dic[biomass_rxn]) <= 1e-6:
            print('Stop growing!')

    return flux_distribution_dict


def write_header(filename, target_reaction):
    fp = open(filename, 'r')
    lines = fp.read()
    fp.close()
    fp = open(filename, 'w')
    fp.write('%s,%s\n'%('reaction', target_reaction))
    fp.write(lines.strip())
    fp.close()
    return

def calculate_MetScore_sum(cobra_model, covariance_data):
    branch_metabolite_data = {}

    for each_metabolite in cobra_model.metabolites:
        tmp_met_id = each_metabolite.id
        branch_metabolite_data[tmp_met_id] = 0.0
        for each_reaction in each_metabolite.reactions:
            reactants = [met.id for met in each_reaction.reactants]
            # products = [met.id for met in each_reaction.products]
            if tmp_met_id in reactants:
                if each_reaction.id in covariance_data:
                    branch_metabolite_data[tmp_met_id] += abs(covariance_data[each_reaction.id])
                    # if (tmp_met_id in reactants and covariance_data[each_reaction.id]>=0):
                    #     branch_metabolite_data[tmp_met_id] += covariance_data[each_reaction.id]
                    # elif tmp_met_id in reactants and covariance_data[each_reaction.id]<0:
                    #     branch_metabolite_data[tmp_met_id] += covariance_data[each_reaction.id]
                    # elif tmp_met_id in products and covariance_data[each_reaction.id]>=0:
                    #     branch_metabolite_data[tmp_met_id] -= covariance_data[each_reaction.id]
                    # else:
                    #     branch_metabolite_data[tmp_met_id] -= covariance_data[each_reaction.id]

    return branch_metabolite_data

def select_candidates(model, target_reaction, output_file,metscore_df,
                      corr_df, cov_df,corr_threshold=0, cov_threshold=0.1):
    pcorr_df = corr_df[corr_df > corr_threshold].dropna()
    pcov_df = cov_df[cov_df > cov_threshold].dropna()
    positive_candidate_reactions = list(set(pcorr_df.index) & set(pcov_df.index))

    ncorr_df = corr_df[corr_df < -corr_threshold].dropna()
    ncov_df = cov_df[cov_df < -cov_threshold].dropna()
    negative_candidate_reactions = list(set(ncorr_df.index) & set(ncov_df.index))

    print('Number of positive candidate reactions: %d' % (len(positive_candidate_reactions)))
    print('Number of negative candidate reactions: %d' % (len(negative_candidate_reactions)))

    fp = open(output_file, 'w')
    header = ['Metabolite', 'Score',
              'No. of reactions', 'No. of positive reactions',
              'No. of negative reactions', 'candidate reactions',
              'positive reactions', 'negative reactions',
              'Positive score', 'Negative score']
    fp.write('%s\n' % ('\t'.join(header)))
    for each_row, each_df in metscore_df.iterrows():
        cobra_metabolite = model.metabolites.get_by_id(each_row)
        candidate_reactions = []

        for each_reaction in cobra_metabolite.reactions:
            for each_reactant in each_reaction.reactants:
                if each_reactant.id == each_row:
                    candidate_reactions.append(each_reaction.id)

        candidate_reactions = list(set(candidate_reactions))
        pos_candidate_reactions = list(set(positive_candidate_reactions) & set(candidate_reactions))
        neg_candidate_reactions = list(set(negative_candidate_reactions) & set(candidate_reactions))

        positive_score = 0.0
        negative_score = 0.0
        for rxn in pos_candidate_reactions:
            positive_score += float(pcov_df.loc[rxn])
        for rxn in neg_candidate_reactions:
            negative_score += float(ncov_df.loc[rxn])

        contents = [each_row, each_df[target_reaction],
                    len(candidate_reactions), len(pos_candidate_reactions),
                    len(neg_candidate_reactions), ';'.join(candidate_reactions),
                    ';'.join(pos_candidate_reactions), ';'.join(neg_candidate_reactions),
                    positive_score, negative_score]
        for i, item in enumerate(contents):
            if i < len(contents) - 1:
                delim = '\t'
            else:
                delim = '\n'
            if type(item) == str:
                fp.write(item + delim)
            else:
                fp.write(str(item) + delim)

    fp.close()
    return


def make_candidate_reaction_sets(df, basename, input_dir, output_dir):
    score_info = {}
    for each_met, each_df in df.groupby('Metabolite'):

        if each_met[-2] == 'c':
            pos_reaction_num = each_df['No. of positive reactions'].values[0]
            neg_reaction_num = each_df['No. of negative reactions'].values[0]

            pos_score = each_df['Positive score'].values[0]
            neg_scroe = each_df['Negative score'].values[0]

            final_pos_scroe = pos_score / np.sqrt(1 + pos_reaction_num)
            final_neg_scroe = neg_scroe / np.sqrt(1 + neg_reaction_num)
            score_info[each_met] = [final_pos_scroe, final_neg_scroe]

    negative_score_info = {}
    positive_score_info = {}

    for met in score_info:
        if abs(score_info[met][0]) > abs(score_info[met][1]):
            positive_score_info[met] = score_info[met][0]
        else:
            negative_score_info[met] = score_info[met][1]

    if not os.path.exists('%s/application_results' % output_dir):
        os.makedirs('%s/application_results' % output_dir)
    fp = open('%s/application_results/Candidate_%s' % (output_dir, basename), 'w')
    fp.write('%s\t%s\t%s\t%s\n' \
             % ('Negative metabolite', 'Positive metabolite',
                'Negative score', 'Positive score'))
    for negative_met in negative_score_info:
        negative_score = negative_score_info[negative_met]

        for positive_met in positive_score_info:
            positive_score = positive_score_info[positive_met]
            fp.write('%s\t%s\t%s\t%s\n' \
                     % (negative_met, positive_met, negative_score, positive_score))

    fp.close()
    return

def metabolite_set(cobra_model):
    met_info = []

    for each_reaction in cobra_model.reactions:
        reactants = [met.id for met in each_reaction.reactants]
        products = [met.id for met in each_reaction.products]
        met_info.append([reactants, products, each_reaction.id, each_reaction.reaction])
    return met_info


def metabolite_info_cytosol(cobra_model):
    met_info_cytosol = []

    for each_reaction in cobra_model.reactions:
        reactants = [met.id for met in each_reaction.reactants]
        products = [met.id for met in each_reaction.products]

        compartments = []
        for each_met in reactants + products:
            each_cmp = each_met[-2]
            compartments.append(each_cmp)

        compartments = list(set(compartments))
        if compartments == ['c']:
            met_info_cytosol.append([reactants, products, each_reaction.id, each_reaction.reaction])

    return met_info_cytosol


def check_up_reaction(filename, met_info, flux_corr_df, flux_cov_df, output_dir):
    up_reaction_list = []
    basename = os.path.basename(filename)
    ex_reaction_id = basename.split('Final_MetScore_')[1].strip()
    ex_reaction_id = ex_reaction_id.replace('.txt', '')

    df3 = pd.read_table(filename)
    if not os.path.exists('%s/application_result2' % output_dir):
        os.makedirs('%s/application_result2' % output_dir)
    fp = open('%s/application_result2/New_reaction_candidate_up_%s' % (output_dir, basename), 'w')
    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
             % ('Target', 'Negative metabolite',
                'Positive metabolite', 'Negative score', 'Positive score',
                'Reaction', 'Equation', 'Corr', 'Cov'))

    for each_row, each_df in df3.iterrows():
        negative_met = each_df['Negative metabolite']
        positive_met = each_df['Positive metabolite']

        negative_score = each_df['Negative score']
        positive_score = each_df['Positive score']

        for each_met_set in met_info:
            if negative_met in each_met_set[0] and positive_met in each_met_set[1]:
                up_reaction_list.append(each_met_set[2])
                target_reaction = each_met_set[2]
                if target_reaction not in flux_corr_df.index:
                    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                             % (ex_reaction_id, negative_met, positive_met,
                                negative_score, positive_score, each_met_set[2],
                                each_met_set[3], 'NA', 'NA'))
                else:
                    corr_val = float(flux_corr_df.loc[target_reaction])
                    cov_val = float(flux_cov_df.loc[target_reaction])
                    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                             % (ex_reaction_id, negative_met, positive_met,
                                negative_score, positive_score, each_met_set[2],
                                each_met_set[3], corr_val, cov_val))
    fp.close()
    return up_reaction_list


def check_down_reaction(filename, met_info, flux_corr_df, flux_cov_df, output_dir):
    down_reaction_list=[]
    basename = os.path.basename(filename)
    ex_reaction_id = basename.split('Final_MetScore_')[1].strip()
    ex_reaction_id = ex_reaction_id.replace('.txt', '')

    df3 = pd.read_table(filename)
    if not os.path.exists('%s/application_result2' % output_dir):
        os.makedirs('%s/application_result2' % output_dir)
    fp = open('%s/application_result2/New_reaction_candidate_down_%s' % (output_dir, basename), 'w')
    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
             % ('Target', 'Negative metabolite',
                'Positive metabolite', 'Negative score', 'Positive score',
                'Reaction', 'Equation', 'Corr', 'Cov'))

    for each_row, each_df in df3.iterrows():
        negative_met = each_df['Negative metabolite']
        positive_met = each_df['Positive metabolite']

        negative_score = each_df['Negative score']
        positive_score = each_df['Positive score']

        for each_met_set in met_info:
            if negative_met in each_met_set[1] and positive_met in each_met_set[0]:
                down_reaction_list.append(each_met_set[2])
                target_reaction = each_met_set[2]
                if target_reaction not in flux_corr_df.index:
                    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                             % (ex_reaction_id, negative_met, positive_met,
                                negative_score, positive_score, each_met_set[2],
                                each_met_set[3], 'NA', 'NA'))
                else:
                    corr_val = float(flux_corr_df.loc[target_reaction])
                    cov_val = float(flux_cov_df.loc[target_reaction])
                    fp.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                             % (ex_reaction_id, negative_met, positive_met,
                                negative_score, positive_score, each_met_set[2],
                                each_met_set[3], corr_val, cov_val))
    fp.close()
    return down_reaction_list