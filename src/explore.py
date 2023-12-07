# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv('../data/train.csv')
data = np.array(df)
np.random.shuffle(data)
data_val = data[:1000].T
data_train = data[1000:].T
y_val = data_val[0]
x_val = data_val[1:]
y_train = data_train[0]
x_train = data_train[1:]
# %%
n0 = 784
n1 = 64
n2 = 10
W1 = np.random.randn(n1,n0)
b1 = np.random.randn(n1,1)
W2 = np.random.randn(n2,n1)
# %%
