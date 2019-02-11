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

frame_length = 2048*8
hop_length = 512*8
eps = 20
min_samples = 160

filename = "./songs/WaitingForLove.mp3"
y, sr = librosa.load(filename, sr=22050)

rmse = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)
X = librosa.segment.recurrence_matrix(rmse)
print("number of points = ", np.count_nonzero(X));
X = np.transpose(np.nonzero(X))

# Compute DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

plt.figure(figsize=(8, 8))
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # white.
        col = [0, 0, 0, 0]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4, markeredgewidth=0.0)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2, markeredgewidth=0.0)

plt.title('Clustering using DBSCAN')
plt.show()


# librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=512*4)
# plt.title('Similarity Matrix')
# plt.show()