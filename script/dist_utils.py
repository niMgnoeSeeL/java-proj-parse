from typing import Tuple
import pandas as pd
import numpy as np

def parse_date(change_hist_df):
    """
    """
    from datetime import datetime
    convert_to_ts = lambda str_chgd_date: datetime.strptime(
        str_chgd_date, '%Y %b %d %H:%M'
        ) if isinstance(str_chgd_date, str) else str_chgd_date#

    change_hist_df.authored_date = change_hist_df.authored_date.apply(convert_to_ts)
    return change_hist_df


def compute_change_vector(
    target_commit: str, 
    change_hist: pd.DataFrame,
    n_min_mod:int = 1) -> Tuple:
    """
    target_commit: 
    """ 
    from hunk_utils import get_changed_method 
    
    target_del_add_mths, _, _ = get_changed_method(change_hist.loc[target_commit].changes)
    if len(target_del_add_mths) <= n_min_mod:
        return (None, None)

    chg_time_vector = {mod_mth:[] for mod_mth in target_del_add_mths}
    chg_commit_vector = {mod_mth:[] for mod_mth in target_del_add_mths}
    tcommit_time = change_hist.loc[target_commit].authored_date 

    # previous changes, INCLUDING the current one 
    upto_change_hist = change_hist.loc[change_hist.authored_date <= tcommit_time]

    #for index_commit, row in tqdm(list(upto_change_hist[['changes', 'authored_date']].iterrows())): 
    for index_commit, row in list(upto_change_hist[['changes', 'authored_date']].iterrows()): 
        deleted_or_added_mths, _, _ = get_changed_method(row.changes)
        intersected_mths = set(target_del_add_mths).intersection(set(deleted_or_added_mths))
        for inter_mth in intersected_mths: 
            chg_time_vector[inter_mth].append(row.authored_date)
            chg_commit_vector[inter_mth].append(index_commit)
    
    return chg_time_vector, chg_commit_vector


def compute_dist(chgdat_of_a, chgdat_of_b, gran = 'sec', _N_recent_time = None):
    """
    dist of a to b
    """
    N_recent_time = -1
    t_N_recent_time = N_recent_time if _N_recent_time is None else _N_recent_time

    if len(chgdat_of_a) == 0 or len(chgdat_of_b) == 0:
        return -1 # meaning nothing has been changeed recently
    else:
        closet_dist = np.min(
            np.abs(
                chgdat_of_a[:,None] - chgdat_of_b),
                axis = 1
            ) 
        t = np.mean(closet_dist).total_seconds() 
        if t_N_recent_time > 0 and t > t_N_recent_time:
            assert False 

        time_dist = np.mean(closet_dist).total_seconds() 
        if gran == 'sec':
            return time_dist 
        elif gran == 'min':
            return np.round(time_dist/60)
        elif gran == 'hour':
            return np.round(time_dist/(60*60))
        elif gran == 'day':
            return np.round(time_dist/(60*60*24))
        else:
            print ("Does not support this granularity ({})".format(gran))
            assert False 