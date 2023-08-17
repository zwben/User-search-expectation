# Organized and restructured code

# 1. Organize imports
import os
import pandas as pd
import numpy as np
import ast
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy import stats
from numpy import median, mean
from statsmodels.stats.multitest import multipletests

# 2. Data Loading and Preprocessing
def load_and_preprocess_data():
    os.chdir(r'H:\\OneDrive - University of Oklahoma\\IR project\\user study\\data')
    
    df_task = pd.read_csv('df_task.csv')
    df_query_act = pd.read_csv('df_query_act.csv')
    df_user = pd.read_csv('df_user.csv')

    df_query_act['TimeFirstClick'] = df_query_act['TimeFirstClick'] / (1000 * 60)
    df_query_act['TimeLastClick'] = df_query_act['TimeLastClick'] / (1000 * 60)
    df_query_act['ScrollDist'] = df_query_act['ScrollDist'] / 1000
    df_query_act.loc[df_query_act['query_order'] == 1, 'NewTerm'] = np.nan
    df_query_act.loc[df_query_act['query_order'] == 1, 'QuerySim'] = np.nan

    return df_task, df_query_act, df_user

# 3. Define helper functions
def add_score(df, score, name):
    dict_score = {'H': name, 'value': score[0], 'p': score[1]}
    df = pd.concat([df, pd.DataFrame(dict_score, index=[0])], ignore_index=True)
    return df

def dync_exp(e, df_temp_nor):
    dync = pd.DataFrame()
    for session in df_temp_nor['session'].unique():
        df_s = df_temp_nor.loc[df_temp_nor['session'] == session]
        dict_s = {
            'session': session,
            'task_type': df_s['task_type_cat'].values[0],
            'task_topic': df_s['task_topic_cat'].values[0],
            'Position': 'First',
            e: df_s[e].loc[df_s[e].first_valid_index()]
        }
        # Assuming pre_exp and last are global variables or constants defined elsewhere
        dict_s = dict_s | dict(zip(pre_exp + last, df_s[pre_exp + last].loc[df_s[e].first_valid_index()]))
        dync = pd.concat([dync, pd.DataFrame(dict_s, index=[0])], ignore_index=True)
        dict_s['Position'] = 'End'
        dict_s[e] = df_s[e].loc[df_s[e].last_valid_index()]
        dync = pd.concat([dync, pd.DataFrame(dict_s, index=[0])], ignore_index=True)
    return dync

def tag_significance(df):
    df['p_sign'] = df['p'].transform(lambda x: '*' if x < 0.05 else '')
    df['correct_p_sign'] = df['correct_p'].transform(lambda x: '*' if x < 0.05 else '')
    return df

# Continuing with the integration of the provided sections:

def perform_additional_statistical_tests(df_temp, df_task, df_user):
    df_p = pd.DataFrame()
    df_p_task = pd.DataFrame()
    df_p_position = pd.DataFrame()

    exp = ['useful_pages', 'clicking_results', 'spending_time']

    # Previous experience
    pre_exp = ['pre_experience', 'familarity', 'difficulty']
    df_temp = df_temp.merge(df_task[pre_exp + ['session', 'user_id_x']], on='session')
    df_temp = df_temp.merge(df_user[['id', 'year']], left_on='user_id_x', right_on='id')
    pre_exp.append('year')

    # In-situ continuity and feedback
    post_exp = ['useful_information', 'effort', 'satisfaction_x']
    off_measure = ['dcg', 'rbp', 'err', 'max_use_score', 'use_number', 'AvgUseScore']
    act_query = ['query_length', 'UniqueTerm', 'NewTerm', 'QuerySim']
    act_mouse = ['click_number', 'Clicks@3', 'Clicks@5', 'Clicks@5+', 'ClickDepth', 'AvgClickRank', 'UniquePage', 'ScrollDist']
    act_time = ['dwell_time_min', 'TimeFirstClick', 'TimeLastClick']

    for s in df_temp['session'].unique():
        for i in post_exp + off_measure:
            df_temp.loc[df_temp['session'] == s, 'last_' + i] = df_temp.loc[df_temp['session'] == s][i].shift(1)

    last = ['last_' + i for i in post_exp + off_measure]

    for i in exp:
        for j in pre_exp + ['query_order'] + last + act_query + act_mouse + act_time:
            score = stats.spearmanr(df_temp[i], df_temp[j], nan_policy='omit')
            df_p = add_score(df_p, score, f'{i}_{j}')
            if score[1] > 0.05:
                for k in ['CE', 'SP', 'RE', 'IP']:
                    score_task = stats.spearmanr(df_temp.loc[df_temp['task_type_cat'] == k][i], df_temp.loc[df_temp['task_type_cat'] == k][j], nan_policy='omit')
                    df_p_task = add_score(df_p_task, score_task, f'{i}_{j}_{k}')

    for i in ['last_useful_information', 'last_effort']:
        for j in act_query + act_mouse + act_time:
            score = stats.spearmanr(df_temp[i], df_temp[j], nan_policy='omit')
            df_p = add_score(df_p, score, f'{i}_{j}')
            if score[1] > 0.05:
                for k in ['CE', 'SP', 'RE', 'IP']:
                    score_task = stats.spearmanr(df_temp.loc[df_temp['task_type_cat'] == k][i], df_temp.loc[df_temp['task_type_cat'] == k][j], nan_policy='omit')
                    df_p_task = add_score(df_p_task, score_task, f'{i}_{j}_{k}')

    for i in exp + ['useful_information', 'effort']:
        score = stats.spearmanr(df_temp[i], df_temp['satisfaction_x'], nan_policy='omit')
        df_p = add_score(df_p, score, f'{i}_satisfaction')
        if score[1] > 0.05:
            for k in ['CE', 'SP', 'RE', 'IP']:
                score_task = stats.spearmanr(df_temp.loc[df_temp['task_type_cat'] == k][i], df_temp.loc[df_temp['task_type_cat'] == k]['satisfaction_x'], nan_policy='omit')
                df_p_task = add_score(df_p_task, score_task, f'{i}_satisfaction_{k}')

    # Assuming the dync_exp function makes use of df_temp_nor
    df_temp_nor = df_temp.copy()
    for i in exp:
        dync = dync_exp(i, df_temp_nor)  # Updated dync_exp to accept df_temp_nor as an argument
        for j in pre_exp:
            for p in ['First', 'End', 'Peak', 'Bottom']:
                score = stats.spearmanr(dync.loc[dync['Position'] == p][i], dync.loc[dync['Position'] == p][j], nan_policy='omit')
                df_p_position = add_score(df_p_position, score, f'{i}_{j}_{p}')

    return df_p, df_p_task, df_p_position

# Main analysis function
def main_analysis():
    df_task, df_query_act, df_user = load_and_preprocess_data()
    
    df_temp = df_query_act.dropna(subset=['useful_pages']).copy()
    df_p, df_p_task = perform_statistical_tests(df_temp)
    
    df_p_additional, df_p_task_additional, df_p_position = perform_additional_statistical_tests(df_temp, df_task, df_user)

    df_p = pd.concat([df_p, df_p_additional], ignore_index=True)
    df_p_task = pd.concat([df_p_task, df_p_task_additional], ignore_index=True)

    return df_p, df_p_task, df_p_position

if __name__ == "__main__":
    df_p, df_p_task, df_p_position = main_analysis()
    df_p.to_csv("results_df_p.csv", index=False)
    df_p_task.to_csv("results_df_p_task.csv", index=False)
    df_p_position.to_csv("results_df_p_position.csv", index=False)