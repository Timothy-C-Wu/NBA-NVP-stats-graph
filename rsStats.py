#import packages
import io
import pandas as pd
import numpy as np
import requests
import sys
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


#def readRSStats():
#open url and read in from table
df = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2019_per_game.html', header= 0, index_col=0, flavor=['lxml'])
'''del df[-1]
#print(df)'''
dfs = pd.concat(df, sort=False)
#print(df)
skipCount = 0
RS_Player = []
RS_index = [-2, -1]
RS_Points = np.array([0.0])
RS_Minutes = np.array([0.0])
RS_Rebounds = np.array([0.0])
RS_WS = np.array([0.0])
RS_totalPoints = np.float64(0.0)
RS_totalMinutes = np.float64(0.0)
RS_totalRebounds = np.float64(0.0)
RS_totalWS = np.float64(0.0)
RS_minPoints = np.float64(0.0)
RS_maxPoints = np.float64(0.0)
RS_minMinutes = np.float64(0.0)
RS_maxMinutes = np.float64(0.0)
RS_minRebounds = np.float64(0.0)
RS_maxRebounds = np.float64(0.0)
RS_minWS = np.float64(0.0)
RS_maxWS = np.float64(0.0)
RS_playerCount = np.int(0)
plt.subplot(1, 1, 1)
for i in range(len(dfs.Player)):
    if(dfs.Player[i] == "Player"):
        skipCount += 1
        continue
    RS_Player.append(dfs.Player[i])
    RS_Minutes = np.insert(RS_Minutes, 0, dfs.MP[i])
    RS_Points = np.insert(RS_Points, 0, dfs.PTS[i])
    RS_Rebounds = np.insert(RS_Rebounds, 0, dfs.TRB[i])
    #statLine = np.stack((RS_Player, RS_Minutes, RS_Points, RS_Rebounds))
    #print (statLine[0], '\n')
    plt.scatter(dfs.MP[i], dfs.PTS[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.PTS[i]), fontsize=8)
    RS_totalPoints += float(dfs.PTS[i])
    RS_totalMinutes += float(dfs.MP[i])
    RS_totalRebounds += float(dfs.TRB[i])
    RS_minPoints = np.amin(RS_Points)
    RS_maxPoints = np.amax(RS_Points)
    RS_minMinutes = np.amin(RS_Minutes)
    RS_maxMinutes = np.amax(RS_Minutes)
    RS_minRebounds = np.amin(RS_Rebounds)
    RS_maxRebounds = np.amax(RS_Rebounds)
    RS_playerCount += 1
#statLine = np.stack((RS_Minutes, RS_Points, RS_Rebounds))
statLineArr = []
p2G = []
p2G.append(input("Please input the name of the player(s) you want to graph.\nTo end input input q.\n\n"))
while(p2G[-1].upper() != "Q"):
    p2G.append(input("Please input the name of the player(s) you want to graph.\nTo end input input q.\n\n"))
p2G.pop(-1)
for i in range(len(p2G)):
    print(p2G[i])
for i in range(len(RS_Points) - 1):
    statLine = []
    statLine.append(RS_Player[i])
    statLine.append(RS_Minutes[i])
    statLine.append(RS_Points[i])
    statLine.append((RS_Rebounds[i]))
    statLineArr.append(statLine)
statLineArr = np.asarray(statLineArr)
#print(len(RS_Player), '\n', '\n')
#print(len(RS_Minutes))
RS_avgPoints = np.float64(RS_totalPoints/RS_playerCount)
RS_avgMinutes = np.float64(RS_totalMinutes/RS_playerCount)
RS_avgRebounds = np.float64(RS_totalRebounds/RS_playerCount)
plt.scatter(RS_avgMinutes, RS_avgPoints, c="blue")
plt.annotate("avg", (RS_avgMinutes, RS_avgPoints), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Points')
plt.title('Points v. Minutes')
plt.show()
for i in range(len(dfs.Player)):
    plt.scatter(dfs.MP[i], dfs.TRB[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.TRB[i]), fontsize=8)
    for x in range(len(p2G)):
        if(dfs.Player[i] == p2G[x]):
            plt.scatter(dfs.MP[i], dfs.TRB[i], c="yellow")
plt.scatter(RS_avgMinutes, RS_avgRebounds, c="blue")
plt.annotate("avg", (RS_avgMinutes, RS_avgRebounds), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Rebounds')
plt.title("Rebounds v. Minutes")
plt.show()
'''for i in range(len(dfs.Player)):
    plt.scatter(dfs.MP[i],dfs.WS[i], c="red")
    plt.annotate(dfs.Player[i], (dfs.MP[i], dfs.WS[i]), fontsize=8)
plt.scatter(RS_avgMinutes, RS_avgWS, c="blue")
plt.annotate("avg", (RS_avgMinutes, RS_avgWS), fontsize=8)
plt.xlabel('Minutes')
plt.ylabel('Win Shares')
plt.title("Win Shares v. Minutes")
plt.show()'''
#print (avgPoints, '\n', avgMinutes)
#df_aggregated = dfs.groupby('Player').mean().reset_index()
'''for i in range(RS_playerCount):
    if(dfs.Player[i] == "Player"):
        continue
    print(RS_Player[i])
    print(RS_Minutes[i])'''