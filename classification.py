import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import json

def get_data_sets(songs):
	samples = []
	for song in songs:
		if song["duration"] == 0:
			print song["song_title"], "has a duration of 0"
		for segment in song["segments"]:
			sample = []
			# sample.append(int(segment["end"]-segment["start"])) # duration
			sample.append(float(segment["start"])/song["duration"]) # relative start
			sample.append(float(segment["energy"])) # energy
			samples.append(sample)
	return np.array(samples)

def get_segment_types(songs):
	sts = []
	for song in songs:
		for segment in song["segments"]:
			st = segment["segment_type"]
			if st == "intro":
				sts.append(0)
			elif st == "verse":
				sts.append(1)
			elif st == "build-up":
				sts.append(2)
			elif st == "drop":
				sts.append(3)
			elif st == "break":
				sts.append(4)
			elif st == "outro":
				sts.append(5)
			else:
				# this shouldn't happen after getting rid of dummy segments
				print "encountered unknown segment type '", st, "' from:", song["song_title"]
				sts.append(6)
	return np.array(sts)

def main():
	# load data from database/data/data.txt
	songs = None
	segments = None
	with open('database/data/data.txt') as json_file:  
		data = json.load(json_file)
		songs = data['songs']
		segments = data['segments']

	# make data set 
	# variables to use for machine learning:
		# length = end - start
		# relative start = start / song_length 
		# energy
		# TODO fullness = computed somehow from stft hopefully
	X = get_data_sets(songs)

	# make target
	y = get_segment_types(songs)

	h = .02  # step size in the mesh
	
	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['red', 'blue', 'green', 'yellow', 'purple', 'orange'])
	n_neighbors = 1
	
	for weights in ["uniform", "distance"]:
		clf = neighbors.KNeighborsClassifier(1, weights=weights)
		clf.fit(X, y)

		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		# x_min x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		# 					 np.arange(y_min, y_max, h))
		# Z = clf.predict()

		# Put the result into a color plot
		# Z = Z.reshape(xx.shape)
		# plt.figure()
		# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
		plt.title("Classification Using Nearest Neightbor")
	
	plt.show()
	# test


main()