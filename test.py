import pandas as pd
import numpy as np
import matplotlib
import sklearn
import statsmodels
import tensorflow as tf
import keras
import pmdarima

print("Версии библиотек:")
print(f"Pandas: {pd.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Statsmodels: {statsmodels.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print(f"Pmdarima: {pmdarima.__version__}")

print("\nПроверка TensorFlow:")
print(tf.config.list_physical_devices())
print(tf.reduce_sum(tf.random.normal([1000, 1000])))