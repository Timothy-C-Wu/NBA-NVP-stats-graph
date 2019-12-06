#import packages
import io
import pandas as pd
import numpy as np
import requests

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

#open url and read in from table
df = pd.read_html('https://www.basketball-reference.com/awards/mvp.html', header= 1, index_col=0, flavor=['lxml'])
del df[-1]
#print(df)
dfs = pd.concat(df, sort=False)
#print(df)
MVP_Player = []
MVP_index = [-2, -1]
MVP_Points = np.array([0.0])
MVP_Minutes = np.array([0.0])
MVP_Rebounds = np.array([0.0])
MVP_WS = np.array([0.0])
MVP_totalPoints = np.float64(0.0)
MVP_totalMinutes = np.float64(0.0)
MVP_totalRebounds = np.float64(0.0)
MVP_totalWS = np.float64(0.0)
MVP_minPoints = np.float64(0.0)
MVP_maxPoints = np.float64(0.0)
MVP_minMinutes = np.float64(0.0)
MVP_maxMinutes = np.float64(0.0)
MVP_minRebounds = np.float64(0.0)
MVP_maxRebounds = np.float64(0.0)
MVP_minWS = np.float64(0.0)
MVP_maxWS = np.float64(0.0)
MVP_playerCount = np.int(0)
plt.subplot(1, 1, 1)
for i in range(len(dfs.Player)):
    MVP_Player.insert(0, dfs.Player[i])
    MVP_Minutes = np.insert(MVP_Minutes, 0, dfs.MP[i])
    MVP_Points = np.insert(MVP_Points, 0, dfs.PTS[i])
    MVP_Rebounds = np.insert(MVP_Rebounds, 0, dfs.TRB[i])
    MVP_WS = np.insert(MVP_WS, 0, dfs.WS[i])
    plt.scatter(dfs.MP[i], dfs.PTS[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.PTS[i]), fontsize=8)
    MVP_totalPoints += dfs.PTS[i]
    MVP_totalMinutes += dfs.MP[i]
    MVP_totalRebounds += dfs.TRB[i]
    MVP_totalWS += dfs.WS[i]
    MVP_minPoints = np.amin(MVP_Points)
    MVP_maxPoints = np.amax(MVP_Points)
    MVP_minMinutes = np.amin(MVP_Minutes)
    MVP_maxMinutes = np.amax(MVP_Minutes)
    MVP_minRebounds = np.amin(MVP_Rebounds)
    MVP_maxRebounds = np.amax(MVP_Rebounds)
    MVP_minWS = np.amin(MVP_WS)
    MVP_maxWS = np.amax(MVP_WS)
    MVP_playerCount += 1
MVP_avgPoints = np.float64(MVP_totalPoints/MVP_playerCount)
MVP_avgMinutes = np.float64(MVP_totalMinutes/MVP_playerCount)
MVP_avgRebounds = np.float64(MVP_totalRebounds/MVP_playerCount)
MVP_avgWS = np.float64(MVP_totalWS/MVP_playerCount)
plt.scatter(MVP_avgMinutes, MVP_avgPoints, c="blue")
plt.annotate("avg", (MVP_avgMinutes, MVP_avgPoints), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Points')
plt.title('Points v. Minutes')
plt.show()
for i in range(len(dfs.Player)):
    plt.scatter(dfs.MP[i], dfs.TRB[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.TRB[i]), fontsize=8)
plt.scatter(MVP_avgMinutes, MVP_avgRebounds, c="blue")
plt.annotate("avg", (MVP_avgMinutes, MVP_avgRebounds), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Rebounds')
plt.title("Rebounds v. Minutes")
plt.show()
for i in range(len(dfs.Player)):
    plt.scatter(dfs.MP[i],dfs.WS[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.WS[i]), fontsize=8)
plt.scatter(MVP_avgMinutes, MVP_avgWS, c="blue")
plt.annotate("avg", (MVP_avgMinutes, MVP_avgWS), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Win Shares')
plt.title("Win Shares v. Minutes")
plt.show()
#print (avgPoints, '\n', avgMinutes)
df_aggregated = dfs.groupby('Player').mean().reset_index()
'''print(df_aggregated)
name = dfs[dfs.Player == 'Player']
name1 = name[name.Player == 35]
points = pd.DataFrame(data =name, )'''

#print(name1)
#print(name)

