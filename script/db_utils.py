## utils 
from typing import Dict, Tuple
from pymongo import MongoClient
from pymongo.database import Database
#import pymongo
from bson.objectid import ObjectId

def get_smartshark_db() -> Database:
    """
    """
    client = MongoClient('localhost', 27017)    
    mydb = client.smartshark_1_1
    return mydb  


def get_hunk(mydb:Database, hunk_id:ObjectId) -> Dict:
    """
    hunk -> seems different for 
    """
    found_hunks = []
    for row in mydb.hunk.find({'_id':hunk_id}):
        found_hunks.append(row)
    
    assert len(found_hunks) == 1
    ret_hunk = found_hunks[0]
    return ret_hunk 


def get_org_lno(mydb:Database, hunk_id:ObjectId) -> Tuple[Dict, Dict]:
    """
    """
    hunk = get_hunk(mydb, hunk_id)

    new_start_lno = hunk['new_start']
    n_new_lines = hunk['new_lines']
    old_start_lno = hunk['old_start']
    n_old_lines = hunk['old_lines']

    if n_new_lines == 0 and n_old_lines == 0:
        return None # nothing is added or deleted 
    
    new_lnos = {cnt:new_start_lno+cnt for cnt in range(n_new_lines)}
    old_lnos = {cnt:old_start_lno+cnt for cnt in range(n_old_lines)}
    return new_lnos, old_lnos
