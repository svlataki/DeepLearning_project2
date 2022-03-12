import tensorflow as tf
import pandas as pd
#A function to generate the dataframe for a csv file
def generate_df(dataset_root, csv_name):
    df = pd.read_csv(dataset_root/csv_name, header=None, names=['filename'])
    df['class'] = (df.filename
                #.str.extract('study.*_(positive|negative)'))
                .str.extract('.*XR_HAND.*study.*_(positive|negative)'))
    return df.dropna()




