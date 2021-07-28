
import pandas as pd
import os
import surprise
import surprise.accuracy
import tqdm
import swifter
import numpy
import statsmodels.formula.api as smf
import swifter
import numpy as np
from sklearn.linear_model import LinearRegression
from formulaic import model_matrix

tqdm.tqdm.pandas()

path = 'C:\\Users\\moehring\\Dropbox (Personal)\\Projects\\PrivacyPenalties\\data'
debug = True
if debug:
    input_dir = os.path.join(path, 'raw', 'MINDsmall_train')
else:
    input_dir = os.path.join(path, 'raw', 'MINDlarge_train')

d = surprise.Dataset.load_builtin('ml-100k')

# read in data
#   behaviors.tsv
#     Impression ID. The ID of an impression. (impression = session)
#     User ID. The anonymous ID of a user.
#     Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
#     History. The news click history (ID list of clicked news) of this user before this impression. The clicked news articles are ordered by time.
#     Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled.

df = pd.read_table(os.path.join(input_dir, 'behaviors.tsv'), sep='\t', header=None,
                   names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

# sample for testing
# uids2keep = np.random.choice(df.user_id.unique(), size=10000, replace=False)
# df = df.loc[df.user_id.isin(uids2keep)].copy()


df['history_len'] = df.history.apply(lambda v: 0 if pd.isna(v) else len(v.split(' ')))
df['impression_len'] = df.impressions.apply(lambda v: 0 if pd.isna(v) else len(v.split(' ')))
num_impressions = df.impression_len.sum()
df.time = pd.to_datetime(df.time)
wide_df = df.copy()

# convert df into long form (each row is an impression / article)
tmp = pd.concat([df[['impression_id', 'user_id', 'time']], df.impressions.str.split(' ', expand=True)], axis=1)
df = pd.melt(tmp, id_vars=['impression_id', 'user_id', 'time'])
df = df.loc[~pd.isna(df.value)]
df = df.rename(columns={'value': 'impression'})
tmp = df.impression.apply(lambda v: v.split('-'))
df['click'] = tmp.apply(lambda v: v[1]).astype(int)
df['article'] = tmp.apply(lambda v: v[0])
df = df.sort_values('time')
assert len(df) == num_impressions

news_df = pd.read_table(os.path.join(input_dir, 'news.tsv'), sep='\t', header=None,
                        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

# join w/ article features
old_len = len(df)
df = pd.merge(left=df, right=news_df, left_on='article', right_on='news_id', how='inner')
assert len(df) == old_len

# drop rare subcategories
# TODO
df = df.sort_values('time')
# sessions = {}
cats = df.subcategory.unique()
user_cat_clicks = {}
user_any_clicks = {}
user_num_impressions_before = {}
for uid, tmp in tqdm.tqdm(df.groupby('user_id')):
    # add individual term
    times = tmp.time.unique()
    user_cat_clicks_inner = {}
    user_any_clicks_inner = {}
    user_num_impressions_before_inner = {}
    for t in times:
        user_num_impressions_before_inner[t] = len(user_any_clicks_inner)
        tmp_t = tmp.loc[(tmp.click == 1) & (tmp.time < t)]
        user_time_click_history = dict((el, 0) for el in cats)
        for el in tmp_t.subcategory:
            user_time_click_history[el] = 1
        user_cat_clicks_inner[t] = user_time_click_history
        user_any_clicks_inner[t] = int(len(tmp_t) > 0)
    user_cat_clicks[uid] = user_cat_clicks_inner
    user_any_clicks[uid] = user_any_clicks_inner
    user_num_impressions_before[uid] = user_num_impressions_before_inner

    # cat_clicks = {}
    # cat_impres = {}
    # u_sessions = {}
    # current_time = None
    # for row in tmp.iterrows():
    #     t = row[1].time
    #     if t != current_time:
    #         u_sessions[t] = dict((el, 0 if cat_clicks[el] == 0 else cat_clicks[el] / cat_clicks[el]) for el in cat_clicks)
    #         current_time = t
    #     cat = row[1]['subcategory']
    #     if cat not in cat_clicks:
    #         cat_clicks[cat] = 0
    #         cat_impres[cat] = 0
    #     cat_clicks[cat] += row[1]['click']
    #     cat_impres[cat] += 1
    # sessions[uid] = u_sessions
#
#
# def individual_term(rr):
#     uuid = rr['user_id']
#     sub_cat = rr['subcategory']
#     time = rr['time']
#     if sub_cat in sessions[uuid][time]:
#         return sessions[uuid][time][sub_cat]
#     return 0


def add_cat_history(rr):
    user_id = rr['user_id']
    time = rr['time'].to_datetime64()
    return user_cat_clicks[user_id][time]


def add_any_click(rr):
    user_id = rr['user_id']
    time = rr['time'].to_datetime64()
    return user_any_clicks[user_id][time]


def add_num_impressions(rr):
    user_id = rr['user_id']
    time = rr['time'].to_datetime64()
    return user_num_impressions_before[user_id][time]


df['click_hist'] = df.progress_apply(add_cat_history, axis=1)
df['clicked'] = df.progress_apply(lambda rr: rr['click_hist'][rr['subcategory']], axis=1)
df['any_click'] = df.progress_apply(add_any_click, axis=1)
df['num_prev_impressions'] = df.progress_apply(add_num_impressions, axis=1)

# split sample into train / test
n = len(df.user_id.unique())
train_n = int(n * 0.8)
train_users = np.random.choice(df.user_id.unique(), size=train_n, replace=False)
train_df = df.loc[df.user_id.isin(train_users)].copy()
test_df = df.loc[~df.user_id.isin(train_users)].copy()

other_cats = train_df.subcategory.value_counts()
other_cats = list(other_cats[other_cats < other_cats.quantile(0.2)].index)
known_cats = [el for el in train_df.subcategory.unique() if el not in other_cats]
train_df.loc[train_df.subcategory.isin(other_cats), 'subcategory'] = 'other'
test_df.loc[~test_df.subcategory.isin(known_cats), 'subcategory'] = 'other'

# "train" SIMPLE model that predicts click based on subcategory
form = 'click ~ 0 + subcategory*clicked'
y, X = model_matrix(form, data=train_df, output='sparse')
mod = LinearRegression(fit_intercept=False, copy_X=False).fit(X=X.todense(), y=y.todense())
# mod = smf.ols('click ~ 0 + subcategory*clicked', data=train_df).fit()
# print(mod.summary())

# now assess quality on test data
_, test_x = model_matrix(form, data=test_df)
test_df['pclick_full'] = mod.predict(test_x)
test_df_private = test_df.copy()
test_df_private['clicked'] = 0
test_df_private['any_click'] = 0
_, test_x = model_matrix(form, data=test_df_private)
test_df['pclick_private'] = mod.predict(test_x)

num_recs = 1
agg_long = pd.DataFrame(index=range(len(df.impression_id.unique())*2),
                        columns=['impression_id', 'user_id', 'topk_clicks', 'total_clicks', 'topk_share',
                                 'algorithm', 'any_click_history']
                        )
row_ix = 0
for iid, tmp in tqdm.tqdm(test_df.groupby('impression_id')):
    total_clicks = tmp.click.sum()
    for el in ['full', 'private']:
        tmp = tmp.sort_values('pclick_{0}'.format(el), ascending=False)
        top_k_clicks = tmp.iloc[0:num_recs].click.sum()
        agg_long.loc[row_ix, 'impression_id'] = iid
        agg_long.loc[row_ix, 'user_id'] = tmp.user_id.iloc[0]
        agg_long.loc[row_ix, 'topk_clicks'] = top_k_clicks
        agg_long.loc[row_ix, 'total_clicks'] = total_clicks
        agg_long.loc[row_ix, 'topk_share'] = top_k_clicks / total_clicks
        agg_long.loc[row_ix, 'algorithm'] = el
        agg_long.loc[row_ix, 'any_click_history'] = tmp.any_click.sum() > 0
        agg_long.loc[row_ix, 'num_prev_impressions'] = tmp.num_prev_impressions.iloc[0]
        row_ix += 1
agg_long.topk_clicks = agg_long.topk_clicks.astype(int)
agg_long['user_id_int'] = agg_long.user_id.progress_apply(lambda v: int(v.replace('U', '')))
first_impressions = {}
first_impression_df = df.loc[(df.num_prev_impressions <= 5) & (df.click == 1)]
for uid, tmp in tqdm.tqdm(first_impression_df.groupby('user_id')):
    cat_counts = tmp.category.value_counts()
    if len(cat_counts) == 0:
        c = 'None'
    else:
        c = cat_counts.index[0]
    first_impressions[uid] = c
agg_long['group'] = agg_long.user_id.apply(lambda vv: first_impressions[vv])

mod2 = smf.ols('topk_clicks ~ algorithm', data=agg_long).fit(cov_type='cluster', cov_kwds={'groups': agg_long['user_id']})
mod3 = smf.ols('topk_clicks ~ algorithm', data=agg_long.loc[agg_long.any_click_history]).fit(cov_type='cluster', cov_kwds={'groups': agg_long.loc[agg_long.any_click_history, 'user_id']})
mod4 = smf.ols('topk_clicks ~ algorithm', data=agg_long.loc[agg_long.num_prev_impressions >= 5]).fit(cov_type='cluster', cov_kwds={'groups': agg_long.loc[agg_long.num_prev_impressions >= 5, 'user_id']})

# let's now analyze accuracy heterogeneity based on first click
mod = smf.ols('topk_clicks ~ 0 + group + algorithm:group', data=agg_long.loc[agg_long.num_prev_impressions >= 5]).fit(cov_type='cluster', cov_kwds={'groups': agg_long.loc[agg_long.num_prev_impressions >= 5, 'user_id']})

# for score in ['pclick', 'pclick_private']:
#     test_df
#
# # let's restrict to sessions with a click
# for el in

#
# # read into surprise format
# reader = surprise.Reader(rating_scale=(0, 1))
#
# # The columns must correspond to user id, item id and ratings (in that order).
# data = surprise.Dataset.load_from_df(df[['user_id', 'article', 'click']], reader)
# trainset = data.build_full_trainset()
# testset = trainset.build_anti_testset()
#
# kf = surprise.model_selection.KFold(n_splits=3)
# algo = surprise.SVD()
# for trainset, testset in kf.split(data):
#
#     # train and test algorithm.
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#
#     # Compute and print Root Mean Squared Error
#     surprise.accuracy.rmse(predictions, verbose=True)
