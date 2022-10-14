import csv
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.utils import check_matplotlib_support

# complete-linkage: d(u, v) = max(dist(u[i],v[j]))

def load_data(filepath):
      with open(filepath, newline='') as csvfile:
        raw_data = csv.DictReader(csvfile)
        data = []
        for row in raw_data:
            data.append(dict((i, row[i]) for i in ('HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed')))
        return data

def calc_features(row):
    value_int = []
    for i in row:
        value_int.append(int(row[i]))
    return np.array(value_int, dtype=np.int64)

def hac(features):
    n = len(features)
    cluster = {}
    for c in range(n):
        cluster[c] = [c]
    distance = [ [0]*(2*n) for i in range(2*n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance[i][j] = abs(np.linalg.norm(features[i] - features[j]))
    Z = np.empty([n-1,4], dtype=float)
    for row in range(n-1):
        min = float("inf")
        for a in cluster:
            for b in cluster:
                dis = distance[a][b]
                if dis != 0 and dis < min:
                    min = dis
                    r0 = a
                    r1 = b
        Z[row][0] = r0
        Z[row][1] = r1
        Z[row][2] = min
        Z[row][3] = len(cluster[a])+len(cluster[b])
        cluster[n+row] = cluster[a] + cluster[b]
        cluster.pop(a, None)
        cluster.pop(b, None)
        for i in cluster:
            cluster[i] = list(set(cluster[i]))
    return Z

def imshow_hac(Z):
    img = hierarchy.dendrogram(Z)
    plt.show()

# hac([calc_features(row) for row in load_data('Pokemon.csv')][:10])
# Z = hierarchy.linkage([calc_features(row) for row in load_data('Pokemon.csv')][:10], method='complete')
# imshow_hac(Z)