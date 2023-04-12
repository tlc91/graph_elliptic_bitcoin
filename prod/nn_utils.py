import numpy as np
import pandas as pd


def remove_nans_from_array(y):
    return y[~np.isnan(y)]

def count_nons_nans_in_array(y):
    return np.count_nonzero(~np.isnan(y))

def nan_padding(x, target_length):
    # padding sequences with nans in order to speed up computation with batches
    # use with caution and propagating nans
    return x + [np.nan] * (target_length - len(x))

def cohort_grouper_fn(df, group_key='cohort_date', 
                   target_variables=['active_users_users'],\
                   group_statics=['dx'], 
                   group_freq='W', 
                   group_function=np.sum):
    return df.groupby([pd.Grouper(key=group_key, freq=group_freq)] + group_statics)[target_variables]\
             .agg(group_function)\
             .reset_index()


def generate_subdomain_input_data(grouped_actual_input, subdomain):

    max_npts =  subdomain.groupby('cohort_date').count().values.max()

    subdomain_input_data = subdomain.merge(grouped_actual_input, on=['cohort_date','dx'], how='left')\
    .groupby('cohort_date')[['dx','retention','cohort_size']]\
    .agg({'dx':list,'retention':list,'cohort_size':min})\
    .assign(npts=lambda x: x['dx'].apply(len),
            retention=lambda x: x['retention'].apply(lambda x: nan_padding(x, max_npts)),
            dx=lambda x: x['dx'].apply(lambda x: nan_padding(x, max_npts))
           )\
    .reset_index()
    
    return subdomain_input_data
