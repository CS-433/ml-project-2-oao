import pandas as pd
import numpy as np

# load the txt files in 'Data/twitter-datasets' into a dataframe
with open('Data/twitter-datasets/train_pos.txt', 'r') as f:
    train_pos = f.readlines()

with open('Data/twitter-datasets/train_neg.txt', 'r') as f:
    train_neg = f.readlines()

# create a dataframe with the text and label
train_df = pd.DataFrame({'text': np.concatenate([train_pos, train_neg]),
                         'label': np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
                         })

# sample at random 0.05 of the rows
train_df = train_df.sample(frac=0.005, random_state=42)

# write the label 1 rows in txt file 't_pos.txt'
with open('Data/twitter-datasets/t_pos.txt', 'w') as f:
    f.writelines(train_df[train_df['label'] == 1]['text'])

# write the label 0 rows in txt file 't_neg.txt'
with open('Data/twitter-datasets/t_neg.txt', 'w') as f:
    f.writelines(train_df[train_df['label'] == 0]['text'])