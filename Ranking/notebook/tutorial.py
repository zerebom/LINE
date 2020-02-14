#!/usr/bin/env python
# coding: utf-8

# In[116]:



from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm_notebook as tqdm

from matplotlib_venn import venn2
from pyspark.ml.recommendation import ALS

get_ipython().system('pip install matplotlib-venn')


# In[117]:



# In[3]:


train_df = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/ua.base',
                       names=["user_id", "item_id", "rating", "timestamp"], sep="\t")


# In[4]:


test_df = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/ua.test',
                      names=["user_id", "item_id", "rating", "timestamp"], sep="\t")


# In[5]:


def set_spark():
    #     globs = pyspark.ml.recommendation.__dict__.copy()

    spark = SparkSession.builder        .master("local[2]")        .appName(
        "ml.recommendation tests")        .getOrCreate()
    return spark


spark = set_spark()


# In[6]:


# df=pd.read_csv('../data/ml-latest-small/ratings.csv')
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=100,
#                                                     stratify=df['userId'])


# In[7]:


def compare(a, b):
    return len(set(a) & set(b))


# In[8]:


item_col = 'item_id'
user_col = 'user_id'
rating_col = 'rating'
df = train_df.copy()

thing_c = CategoricalDtype(
    sorted(df[item_col].unique()), ordered=True)
person_c = CategoricalDtype(
    sorted(df[user_col].unique()), ordered=True)

row = df[item_col].astype(thing_c).cat.codes
col = df[user_col].astype(person_c).cat.codes

matrix = csr_matrix((df[rating_col], (row, col)),
                    shape=(thing_c.categories.size, person_c.categories.size))

items = sorted(df[item_col].unique())
uids = sorted(df[user_col].unique())


# In[9]:


def _indptr2row(matrix):
    rows = []
    for i in range(len(matrix.indptr)):
        mini_list = [i] * (matrix.indptr[i] - matrix.indptr[i - 1])
        rows.extend(mini_list)
    return rows


rows = _indptr2row(matrix)


# In[10]:


rank = 10
seed = 0
NMF = True
iter = 5


# ### ALS

# In[272]:


rdd = [Row(user_id=int(i) + 1, item_id=int(u), data=int(d))
       for u, i, d in zip(rows, matrix.indices, matrix.data)]
ratings = spark.createDataFrame(rdd)


print(rank, iter, seed, NMF)

# 特徴量の次元数↓
als = ALS(rank=rank, maxIter=iter, seed=seed, nonnegative=True, implicitPrefs=True, regParam=0.01,
          userCol="user_id", itemCol="item_id", ratingCol="data", coldStartStrategy='drop')
model = als.fit(ratings)

out_df = model.recommendForAllUsers(300)


# In[274]:


out_pdf = out_df.toPandas()


# In[275]:


def _parse_data(x, col='item_id'):
    return [int(row[col]) for row in x]


out_pdf['recommendations_id'] = out_pdf['recommendations'].apply(_parse_data)
out_pdf2 = out_pdf[['user_id', 'recommendations_id']]


# In[282]:


als_pred = out_pdf2.sort_values('user_id')['recommendations_id'].values


als_pred = np.array([list(set(als_pred[i]) - set(train_df[train_df['user_id'] == i + 1]
                                                 ['item_id'].values))[:10] for i in range(len(als_pred))])


# als_pred=np.array([i for i in range(als_pred)])
# 0-indexなので
# als_pred+=1


# In[283]:


np.mean([len(compare(sorted(train_df[train_df['user_id'] == i + 1]['item_id'].values), als_pred[i]))
         for i in range(len(item_knn_pred))])


# In[284]:


np.mean([len(compare(sorted(test_df[test_df['user_id'] == i + 1]['item_id'].values), als_pred[i]))
         for i in range(len(item_knn_pred))])


# ### user_knn

# In[95]:


rated_arr = matrix.copy()
rated_arr[rated_arr != 0] = 1
rated_arr = np.abs(rated_arr.toarray() - 1)


user_knn_pred = []
for uid in tqdm(range(matrix.shape[1])):
    # user*user similarity matrix
    user_sim_mat = cosine_similarity(matrix.T, dense_output=False)

    most_sim_user = user_sim_mat[uid].toarray().argsort()[0][::-1][1:6]
    most_sim_val = user_sim_mat[uid, sim_user].toarray()

    # sum(similarity*score)
    harmonic_ave_score = matrix[:, sim_user].multiply(most_sim_val).sum(axis=1)

    # get idx & fix 1-index
    pred = np.argsort(np.ravel(harmonic_ave_score) * rated_arr[:, uid])[::-1][:10] + 1
    user_knn_pred.append(pred)

user_knn_pred = np.array([i for i in user_knn_pred])

user_knn_pred


# In[97]:


np.mean([compare(sorted(train_df[train_df['user_id'] == i + 1]['item_id'].values), user_knn_pred[i])
         for i in range(len(user_knn_pred))])


# In[98]:


np.mean([compare(sorted(test_df[test_df['user_id'] == i + 1]['item_id'].values), user_knn_pred[i])
         for i in range(len(user_knn_pred))])


# ### Item_knn

# In[107]:


rated_arr = matrix.copy()
rated_arr[rated_arr != 0] = 1
rated_arr = np.abs(rated_arr.toarray() - 1)

sim_arr = cosine_similarity(matrix, dense_output=False)

item_knn_pred = []
for uid in tqdm(range(matrix.shape[1])):
    # ユーザが評価したアイテムのrating
    user_rating_vec = matrix[:, uid].T

    # simirality*rating
    pred_rating_vec = sim_arr.multiply(user_rating_vec)

    # delete rated items
    pred_user_item = np.ravel(pred_rating_vec.sum(axis=1)) * rated_arr[:, uid]

    # get idx & fix 1-index
    pred = np.argsort(pred_user_item)[::-1][:10] + 1

    item_knn_pred.append(pred)

item_knn_pred = np.array([i for i in item_knn_pred])


# In[108]:


np.mean([compare(sorted(train_df[train_df['user_id'] == i + 1]['item_id'].values), item_knn_pred[i])
         for i in range(len(item_knn_pred))])


# In[109]:


np.mean([compare(sorted(test_df[test_df['user_id'] == i + 1]['item_id'].values), item_knn_pred[i])
         for i in range(len(item_knn_pred))])


# ### eda
# - coverage
# - accracy
# -

# ### mrr

# In[ ]:


ans_arr = np.array([list(test_df[test_df['user_id'] == i + 1]['item_id'].values) for i in range(len(item_knn_pred))])


# In[114]:


def calc_mrr(rank_list):
    mrr_list = []
    for mrr in rank_list:
        if mrr != -1:
            mrr_list.append(1 / mrr)
        else:
            mrr_list.append(0)
    print(np.mean(mrr_list))


# In[115]:


rank_list = []
preds = [item_knn_pred, user_knn_pred]

for pred in preds:
    for i in range(len(pred)):
        if sum(np.isin(ans_arr[i], pred[i])) != 0:
            rank_list.append(np.nonzero(np.isin(ans_arr[i], pred[i]))[0][0] + 1)
        else:
            rank_list.append(-1)
    plt.hist(rank_list, bins=11)
    calc_mrr(rank_list)

    plt.show()


# ## coverage

# In[125]:


plt.rcParams["font.size"] = 18


user = set(np.ravel(user_knn_pred))
item = set(np.ravel(item_knn_pred))
both = len(user & item)
user = len(user)
item = len(item)


venn2(subsets=(user - both, item - both, both), set_labels=('user_knn', 'item_knn'))
plt.show()


# ### hit_ratio

# In[145]:


hit_item_knn = [list(set(ans_arr[i]) & set(item_knn_pred[i])) for i in range(len(item_knn_pred))]
hit_user_knn = [list(set(ans_arr[i]) & set(user_knn_pred[i])) for i in range(len(user_knn_pred))]


def flatten(_list):
    new_list = []
    for val in _list:
        new_list.extend(val)
    return np.unique(new_list)


hit_item_knn = flatten(hit_item_knn)
hit_user_knn = flatten(hit_user_knn)

user = set(np.ravel(hit_user_knn))
item = set(np.ravel(hit_item_knn))
both = len(user & item)
user = len(user)
item = len(item)


venn2(subsets=(user - both, item - both, both), set_labels=('user_knn', 'item_knn'))
plt.show()


# ### ndcg

# In[ ]:


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


# In[147]:


# user_id,item_id両方をkeyとしてratingを取得した
ndcg_df = test_df.copy()


def zfill(x):
    return str(x).zfill(4)


ndcg_df['con_id'] = ndcg_df['user_id'].apply(zfill) + ndcg_df['item_id'].apply(zfill)
ratings = ndcg_df[['rating', 'con_id']].set_index('con_id').to_dict()['rating']


# In[156]:


def calc_ndcg(preds, ratings, rank=10):
    encodes, ndcgs = [], []
    for i in range(len(preds)):
        for j in range(10):
            encodes.append(zfill(i + 1) + zfill(preds[i, j]))
    for encode in encodes:
        if encode in ratings:
            ndcgs.append(ratings[encode])
        else:
            ndcgs.append(0)
    return np.array(ndcgs).reshape(-1, rank)


item_rank = calc_ndcg(item_knn_pred, ratings)
user_rank = calc_ndcg(user_knn_pred, ratings)


# In[173]:


print(np.mean([ndcg_at_k(rating, 10) for rating in item_rank]))
print(np.mean([ndcg_at_k(rating, 10) for rating in user_rank]))


# ### jini

# In[217]:


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient


# In[218]:


item_candidates = test_df['item_id'].nunique()


# In[223]:


fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
preds = {'item_pred': item_knn_pred, 'user_knn': user_knn_pred}
ginis = []
for i, (name, pred) in enumerate(preds.items()):
    _, item_counts = np.unique(np.ravel(pred), return_counts=True)
    pad = np.zeros(item_candidates - len(item_counts))
    item_counts = np.append(pad, item_counts)
    stand_gini = np.cumsum(np.sort(item_counts)) / sum(item_counts)
    ginis.append(round(gini(stand_gini), 3))

    x = np.linspace(0, 1, item_candidates)
    ax.plot(x, stand_gini, label=name)


ax.plot(x, x, label='equal distribution')
ax.legend(loc=2)
ax.set_title(f'gini = item:{ginis[0]} user:{ginis[1]}')
fig.show()


# In[203]:


# In[205]:


plt.plot(x, stand_jini)
plt.plot(x, x)


# In[293]:


plt.hist(np.ravel(test_data), bins='auto', label='test')
# plt.hist(np.ravel(als_pred),bins='auto',label='als')
plt.hist(np.ravel(item_knn_pred), bins='auto', label='knn')

plt.legend()
plt.show()


# In[ ]:


# 最頻出アイテムがどれくらいかぶっているか
a, b = np.unique(als_pred, return_counts=True)
ddd = pd.DataFrame([a, b]).T.sort_values(1, ascending=False).set_index(0).head(30).index
dd = pd.DataFrame(test_df['item_id'].value_counts().head(30)).index


# In[ ]:


plt.scatter(idxs, np.array(sim_arr.sum(axis=0))[0])
parr = np.array(pred_rate_arr)[0]
idxs = list(range(pred_rate_arr.shape[1]))

plt.scatter(idxs, parr)


# In[215]:


plt.figure(figsize=(12, 12))
repeat_idx = np.repeat(np.array(range(len(knn_pred))), 10)
ans_data = np.ravel(knn_pred)
plt.scatter(repeat_idx, ans_data)


# In[214]:


plt.figure(figsize=(12, 12))
repeat_idx = np.repeat(np.array(range(len(als_pred))), 10)
ans_data = np.ravel(als_pred)
plt.scatter(repeat_idx, ans_data)


# In[213]:


plt.figure(figsize=(12, 12))
repeat_idx = np.repeat(np.array(range(len(test_data))), 10)
ans_data = np.ravel(test_data)
plt.scatter(repeat_idx, ans_data)


# In[ ]:


class Item_knn(object):
    def __init__(self, top_k, parallel, items, matrix):
        self.top_k = top_k
        self.parallel = parallel
        self.similarities = cosine_similarity(matrix, dense_output=False)
        self.matrix = matrix
        self.items = items

        self.rated_arr = matrix.T.copy()
        self.rated_arr[self.rated_arr != 0] = 1
        self.rated_arr = np.abs(self.rated_arr.toarray() - 1)

    def search_nearest_item_ids(self, index):
        predicted_vector = np.array(self.similarities[index].todense())

        # 自分自身は0にする
        predicted_vector[0][index] = 0
        # 評価済みのアイテムの予測値を0にする
        predicted_vector = predicted_vector * self.rated_arr[index]

        # コサイン類似度を降順に並べる
        recommended_item = [i for i in np.argsort(predicted_vector)[0][::-1][:self.top_k]]
        # その商品をとってきて並べる
        recommended_item2 = [self.items[f] for f in recommended_item]
        return recommended_item2

    # レコメンドするときにそれぞれでマルチプロセスを回せるようにするもの。
    def __sub_process(self, queue, part):

        ini = int(len(self.items) / self.parallel * part)
        fin = int(len(self.items) / self.parallel * (part + 1))
        results = [
            (self.items[i], self.search_nearest_item_ids(i)) for i in range(ini, fin)]
        queue.put(results)

    def main(self):
        t0 = time()
        q = mp.Queue()
        ps = [mp.Process(target=self.__sub_process, args=(q, i)) for i in range(self.parallel)]
        [p.start() for p in ps]
        results = [q.get() for _ in range(self.parallel)]
        results = list(chain.from_iterable(results))
        results = [[t_id, self.items] for t_id, self.items in results if
                   self.items[0] != ""]  # Delete item whose recommend item is None.
        print("%.3f sec" % (time() - t0))
        return results
