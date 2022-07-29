"""
Compute 
"""
from typing import Dict, List, Tuple
from git import Object
import hunk_utils
import db_utils
import dist_utils 
import change_utils
import pandas as pd 
from pymongo.database import Database
from tqdm import tqdm 

datadir = "../data/parsed"

class MethodLabelInfo(Object):
    def __init__(self, 
        mth_key:Tuple[str,str], 
        mth_label:str, 
        file:str, 
        new_lno_labels:Dict, 
        old_lno_labels:Dict,
        authorship:str):
        super().__init__()
        
        self.mth_key = mth_key 
        self.file = file 
        self.mth_label = mth_label 
        self.new_lno_labels = new_lno_labels
        self.old_lno_labels = old_lno_labels
        self.authorship = authorship
        # related to 
        self.file = file 

        # key = line_no,
        # value = (hunk_id, index)
        self.chgdat_ts_arr = None # change time vector
        self.chgdat_rev_hash_arr = None # change commit hash vector


    def set_chgdat_arr(self,):
        """
        """
        self.inex = ...
        

    def set_chgdat_rev_arr(self):
        """
        """
        pass 


#def get_final_mths_labels(
    #mydb:Database,
    #hunk_df:pd.DataFrame,
    #change_hist_df:pd.DataFrame,
    #project:str, 
    #target_commits:List[str],
    #simple:bool = True) -> Dict[str,str]:
#
    #final_labels_pc = {}
    #for commit in tqdm(target_commits):
        #hunk_rows = hunk_utils.get_target_rows(hunk_df, project, commit)
        #new_mths_w_labels = {}
        #old_mths_w_labels = {}
        #for _, hunk_row in hunk_rows.iterrows():
            ## check whether it modified javafiles
            #filepath = hunk_row.file 
            #if not filepath.endswith('.java'):
                #continue 
#
            #hunk_id = hunk_row.hunk_id 
            #new_old_lnos = db_utils.get_org_lno(mydb, hunk_id)
            #if new_old_lnos is None:
                #continue
            #else:
                #lno_labels = hunk_utils.get_all_labels(hunk_df, project, hunk_id)
#
                ## for new lines
                #new_lnos, old_lnos = new_old_lnos
                ## should consider the thing that, sometims, idx_new_lno not labeled
                #for idx_new_lno, org_new_lno in new_lnos.items():
                    #new_mth = hunk_utils.get_belong_chgd_mth(
                        #change_hist_df.loc[commit].changes, 
                        #filepath, org_new_lno, False)
#
                    #if new_mth is not None:
                        ## get label
                        #try:
                            #man_lno_label = lno_labels[idx_new_lno]
                        #except KeyError: #idx_new_lno not labeled
                            #continue # skip this 
#
                        #try:
                            #new_mths_w_labels[new_mth].append(man_lno_label)
                        #except KeyError:
                            #new_mths_w_labels[new_mth] = [man_lno_label]
#
                ## for old lines
                #for idx_old_lno, org_old_lno in old_lnos.items():
                    #old_mth = hunk_utils.get_belong_chgd_mth(
                        #change_hist_df.loc[commit].changes, 
                        #filepath, org_old_lno, True)
                    #if old_mth is not None:
                        ## get label
                        #try:
                            #man_lno_label = lno_labels[idx_old_lno]
                        #except KeyError: #idx_new_lno not labeled (can happen)
                            #continue # skip this 
                        #try:
                            #old_mths_w_labels[old_mth].append(man_lno_label)
                        #except KeyError:
                            #old_mths_w_labels[old_mth] = [man_lno_label]
#        
        ## up-to-here, new_mths_w_labels & old_mths_w_labels -> key:mth, 
        ## determine the final label of modified methods 
        #final_mths_labels = hunk_utils.get_final_labels(
            #new_mths_w_labels, old_mths_w_labels, simple = simple)
        #final_labels_pc[commit] = final_mths_labels
#    
    #return final_labels_pc


def get_mths_labels(
    mydb:Database,
    hunk_df:pd.DataFrame,
    change_hist_df:pd.DataFrame,
    project:str, 
    target_commits:List[str]) -> Dict[str,str]:
    """
    return a dictionary of method label information computed per line
    """
    from tqdm import tqdm
    mth_labels_pc = {}
    for commit in target_commits:
        hunk_rows = hunk_utils.get_target_rows(hunk_df, project, commit)
        new_mths_w_labels = {}
        old_mths_w_labels = {}
        for _, hunk_row in hunk_rows.iterrows():
            # check whether it modified javafiles
            filepath = hunk_row.file 
            if not filepath.endswith('.java'):
                continue 
            
            hunk_id = hunk_row.hunk_id 
            new_old_lnos = db_utils.get_org_lno(mydb, hunk_id)
            if new_old_lnos is None:
                continue
            else:
                lno_labels = hunk_utils.get_all_labels(hunk_df, project, hunk_id)
                # for new lines
                new_lnos, old_lnos = new_old_lnos
                # should consider the thing that, sometims, idx_new_lno not labeled
                for idx_new_lno, org_new_lno in new_lnos.items():
                    new_mth = hunk_utils.get_belong_chgd_mth(
                        change_hist_df.loc[commit].changes, 
                        filepath, org_new_lno, False)
                    if new_mth is not None:
                        # get label
                        try:
                            man_lno_label = lno_labels[idx_new_lno]
                        except KeyError: #idx_new_lno not labeled
                            continue # skip this 
                        try:
                            new_mths_w_labels[new_mth][org_new_lno] = man_lno_label
                        except KeyError:
                            new_mths_w_labels[new_mth] = {org_new_lno:man_lno_label}

                # for old lines
                for idx_old_lno, org_old_lno in old_lnos.items():
                    old_mth = hunk_utils.get_belong_chgd_mth(
                        change_hist_df.loc[commit].changes, 
                        filepath, org_old_lno, True)
                    if old_mth is not None:
                        # get label
                        try:
                            man_lno_label = lno_labels[idx_old_lno]
                        except KeyError: #idx_new_lno not labeled (can happen)
                            continue # skip this 
                        try:
                            old_mths_w_labels[old_mth][org_old_lno] = man_lno_label
                        except KeyError:
                            old_mths_w_labels[old_mth] = {org_old_lno:man_lno_label}
        #
        mth_labels_pc[commit] =  {'old':old_mths_w_labels, 'new':new_mths_w_labels}
    return mth_labels_pc


def get_target_commits(
    repo_path:str, 
    target_cs_file:str = None) -> List:
    """
    will be used to .... 
    """
    target_commits = pd.read_csv(target_cs_file).commit.values
    repo_path = os.path.join(repo_path)
    repo = git.Repo(repo_path)
    all_commits = list(repo.iter_commits()) 
    all_commits = [c for c in all_commits if c.hexsha in target_commits]
    return all_commits


def get_labeled_mths(
    project:str, 
    mydb:Database,
    changes_df,
    hunk_df:pd.DataFrame, 
    simple:bool = True,
    target_commit_insts:List = None) -> Dict:
    """
    will compute change vector 
    """
    # retrieve an entire list of chgd_time_vector 
    final_mth_labels_pc = {}
    for commit_inst in tqdm(target_commit_insts):
        author = commit_inst.author.name 
        commit = commit_inst.hexsha
        
        # line-level labels 
        mth_labels_pl = get_mths_labels(
            mydb,
            hunk_df, changes_df,
            project, [commit])[commit]

        # decide the final label of individual method modified in a commit
        final_mth_labels = hunk_utils.get_final_labels(
            mth_labels_pl['new'], 
            mth_labels_pl['old'], 
            simple = simple)
        
        final_mth_labels_pc[commit] = {'author':author} 
        if len(final_mth_labels) > 0:
            for mod_mth in final_mth_labels.keys():
                final_mth_labels_pc[commit][mod_mth] = {
                    'mth_label':final_mth_labels[mod_mth], 
                    'new_lnos':None, 
                    'old_lnos':None}
                try:
                    final_mth_labels_pc[commit][mod_mth]['new_lnos'] = mth_labels_pl['new'][mod_mth]
                except KeyError:
                    pass 
                try:
                    final_mth_labels_pc[commit][mod_mth]['old_lnos'] = mth_labels_pl['old'][mod_mth]
                except KeyError:
                    pass 
        else:
            mod_mths = list(set(
                list(mth_labels_pl['new'].keys()) + list(mth_labels_pl['old'].keys())))
            for mod_mth in mod_mths: 
                final_mth_labels_pc[commit][mod_mth] = {
                    'mth_label':None, 'new_lnos':None, 'old_lnos':None}
                try:
                    final_mth_labels_pc[commit][mod_mth]['new'] = mth_labels_pl['new'][mod_mth]
                except KeyError:
                    pass 
                try:
                    final_mth_labels_pc[commit][mod_mth]['old'] = mth_labels_pl['old'][mod_mth]
                except KeyError:
                    pass 

    return final_mth_labels_pc      


def add_chgdat_vector(
    mod_mths_info, 
    commit:str, 
    change_hist: pd.DataFrame) -> Dict:
    """
    """
    # chgd_time_vector -> from parsing the proj repo 
    chgd_time_vector ,_ = dist_utils.compute_change_vector(commit, change_hist)
    for k in mod_mths_info.keys():
        if k == 'author':
            continue
        mod_mths_info[k]['chgdat'] = chgd_time_vector[k] # keyerror? -> not sure
    return mod_mths_info


def compute_distances(
    which_types:List[str],
    mod_mths_info:pd.DataFrame) -> Dict[str,float]:
    """
    which_types -> chgdat, author, lexical sim 
    """
    import numpy as np
    from itertools import combinations
    
    ret_dists = {}
    mod_mths = list(mod_mths_info.keys())
    mod_mths.remove('author')
    mth_pairs = list(combinations(mod_mths, 2))
    #author = mod_mths_info['author']
    for a_mod_mth, b_mod_mth in mth_pairs:
        a_info = mod_mths_info[a_mod_mth]
        b_info = mod_mths_info[b_mod_mth]
        ret_dists[(a_mod_mth, b_mod_mth)] = {}
        for which_type in which_types:
            if which_type == 'chgdat':
                a_dist_arr = np.array(a_info['chgdat'])
                a_dist_arr.sort()
                a_dist_arr = a_dist_arr[::-1]

                b_dist_arr = np.array(b_info['chgdat'])
                b_dist_arr.sort()
                b_dist_arr = b_dist_arr[::-1]

                a_to_b_dist = dist_utils.compute_dist(
                    a_dist_arr[1:], b_dist_arr, gran = 'day')
                b_to_a_dist = dist_utils.compute_dist(
                    b_dist_arr[1:], a_dist_arr, gran = 'day')

                ret_dists[(a_mod_mth, b_mod_mth)][
                    which_type] = [a_to_b_dist, b_to_a_dist]
            elif which_type == 'irfl':
                pass 
            elif which_type == 'authorship':
                pass

    return ret_dists


def format():
    """
    """
    pass 


if __name__ == "__main__":
    import git, os

    with_chgdat_v = True
    with_compute_dist = False 

    project = 'ant-ivy' 
    repo_path = os.path.join(
        '/Volumes/JJ_Media/Data/commit_untangling/projects', project)
    target_cs_file = "../data/cands/ant_ivy.csv"
    target_commit_insts = get_target_commits(
        repo_path, target_cs_file=target_cs_file)
    
    # get hunks
    hunk_file = "../data/complete_hunk_labels.pkl"
    completed_df = pd.read_pickle(hunk_file)
    changes_df = change_utils.get_change_df(project, datadir = "../data/parsed")

    # assume mongodb to run & smartshark_1_1 database to be already restored 
    target_commits = [] # ...
    mydb = db_utils.get_smartshark_db()
    labeled_mths_pc = get_labeled_mths(
        project, 
        mydb, 
        changes_df,
        completed_df,
        simple = True, 
        target_commit_insts = target_commit_insts)
    
    os.makedirs("../data/cands", exist_ok=True)
    destfile = "../data/cands/ant_ivy_labeled_mths.pkl"
    with open(destfile, 'wb') as f:
        import pickle
        pickle.dump(labeled_mths_pc, f)

    #if with_chgdat_v:
        #for commit, labeled_mths in tqdm(labeled_mths_pc.items()) :
            #labeled_mths_pc[commit] = add_chgdat_vector(
                #labeled_mths, commit, changes_df)
    
    # compute distance
    #if with_compute_dist:
    #    for commit, mod_mths_info in labeled_mths_pc.items():
    #        pairwise_dists = compute_distances(['chgdat'], mod_mths_info)
            

