import argparse
import numpy as np
import pandas as pd

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def npStackWithNone(allData,newData,prepend=None):
    if prepend is None:
        return newData if allData is None else np.vstack((allData, newData))
    else:
        return np.array([str(prepend),newData],dtype=object) if allData is None else np.vstack((allData,np.array([str(prepend),newData],dtype=object)))

def pdStackWithNone(allData,newData,prepend=None):
    if prepend is None:
        print("prepend needs to be defined when stacking dataFrames!")
        exit(0)
    else:
        df_temp = newData
        if not "ID" in df_temp.columns:
            df_temp["ID"] = int(prepend)
        return df_temp if allData is None else pd.concat((allData,newData),ignore_index=True)