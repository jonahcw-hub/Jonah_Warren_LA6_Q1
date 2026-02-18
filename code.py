import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data = pd.read_csv(url, header=None)

data.columns = [
    "class","cap-shape","cap-surface","cap-color","bruises","odor",
    "gill-attachment","gill-spacing","gill-size","gill-color",
    "stalk-shape","stalk-root","stalk-surface-above-ring",
    "stalk-surface-below-ring","stalk-color-above-ring",
    "stalk-color-below-ring","veil-type","veil-color","ring-number",
    "ring-type","spore-print-color","population","habitat"
]

transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'eggs'],
    ['milk', 'butter'],
    ['milk', 'bread', 'eggs', 'butter']
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
small_df = pd.DataFrame(te_array, columns=te.columns_)

one_hot = pd.get_dummies(data)

min_support = 500 / len(one_hot)
small_support = 0.3

#Apriori Mushroom
start = time.time()
apriori_result = apriori(one_hot, min_support=min_support, use_colnames=True, low_memory=True)
apriori_time = time.time() - start

#FP-Growth Mushroom
start = time.time()
fpgrowth_result = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
fpgrowth_time = time.time() - start

#Apriori Small_data
start = time.time()
small_ap = apriori(small_df, min_support=small_support, use_colnames=True, low_memory=True)
small_ap_time = time.time() - start

#FP-Growth Small_data
start = time.time()
small_fp = fpgrowth(small_df, min_support=small_support, use_colnames=True)
small_fp_time = time.time() - start

print("\n--- Mushroom Results ---")
print("Apriori Time:", apriori_time)
print("FP-Growth Time:", fpgrowth_time)
print("Apriori Itemsets:", len(apriori_result))
print("FP-Growth Itemsets:", len(fpgrowth_result))

print("\n--- Small_data Results ---")
print("Apriori:", small_ap_time)
print("FP-Growth:", small_fp_time)