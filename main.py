#import packages
import io
import pandas as pd
import numpy as np
import requests
import sys
import rsStats
import mvpStats

np.set_printoptions(threshold=sys.maxsize)


#to plot within notebook
import matplotlib
import matplotlib.pyplot as plt


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

#for nomalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

