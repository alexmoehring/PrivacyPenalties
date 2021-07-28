
import pandas as pd
import NewsSocialSignaling
import os
import pickle
import numpy as np
import tqdm
import ResearchTools

twitter_path = 'C:/Users/moehring/Dropbox (Personal)/Projects/NewsSocialSignaling/data/raw/newspaper_quality'
fn = os.path.join(twitter_path, 'newspaper_quality_processed_all_pubs.txt')

pubs = pd.read_table(fn, sep='\t')
print(pubs.head())

# load random sample
sample_name = 'random_sample_2013'

tmp_fn = 'C:/users/moehring/Downloads/tmp_twitters.p'
if True and os.path.exists(tmp_fn):
    with open(tmp_fn, 'rb') as f:
        existing_users = pickle.load(f)
else:
    existing_users = [el for el in NewsSocialSignaling.Database.get_all_users()
                      if isinstance(el['extra_data'], dict)
                      and 'sample_name' in el['extra_data']
                      and el['extra_data']['sample_name'] == sample_name
                      and not el['most_recent'].protected
                      ]
    with open(tmp_fn, 'wb') as f:
        pickle.dump(existing_users, f)

# ingest into user / publisher matrix
assert len(pubs.single_twitter_handle) == len(pubs.single_twitter_handle.unique())
df = pd.DataFrame(index=[el['id'] for el in existing_users], columns=pubs.single_twitter_handle)

pubid2twitter = dict((row[1]['twitter_id'], row[1]['single_twitter_handle']) for row in pubs.iterrows())
for user in tqdm.tqdm(existing_users):
    friends = user['most_recent_friends']
    pub_friends = [el for el in friends if el in pubid2twitter]
    for p in pub_friends:
        df.loc[user['id'], pubid2twitter[p]] = 1

# first plot distribution of who follows who
pub_shares = df.fillna(0).mean()


# form train / test sets
rnd = np.random.RandomState(seed=809034)
train_share = 0.9
train = rnd.choice(existing_users, size=int(len(existing_users) * train_share))
