import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth

# Load mushroom dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
data = pd.read_csv(url, header=None)

# Add column names
data.columns = [
    "class","cap-shape","cap-surface","cap-color","bruises","odor",
    "gill-attachment","gill-spacing","gill-size","gill-color",
    "stalk-shape","stalk-root","stalk-surface-above-ring",
    "stalk-surface-below-ring","stalk-color-above-ring",
    "stalk-color-below-ring","veil-type","veil-color","ring-number",
    "ring-type","spore-print-color","population","habitat"
]

# Convert to one-hot encoded transaction format
one_hot = pd.get_dummies(data)

# Convert min_support = 500 transactions into fraction
min_support = 500 / len(one_hot)

#Apriori
start = time.time()
apriori_result = apriori(one_hot, min_support=min_support, use_colnames=True, low_memory=True)
apriori_time = time.time() - start

#FP-Growth
start = time.time()
fpgrowth_result = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
fpgrowth_time = time.time() - start

# Print results
print("\n--- Results ---")
print("Apriori Time:", apriori_time)
print("FP-Growth Time:", fpgrowth_time)
print("Apriori Itemsets:", len(apriori_result))
print("FP-Growth Itemsets:", len(fpgrowth_result))