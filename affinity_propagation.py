# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from itertools import cycle

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

filename = "./songs/WaitingForLove.mp3"
y, sr = librosa.load(filename, sr=22050)

rmse = librosa.feature.rmse(y=y, frame_length=2048*8, hop_length=512*4)
X = librosa.segment.recurrence_matrix(rmse)
X = np.transpose(np.nonzero(X))

print("number of points = ", np.count_nonzero(X));
# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(max_iter=10, convergence_iter=3).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
# #############################################################################
# Plot result

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=1)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()