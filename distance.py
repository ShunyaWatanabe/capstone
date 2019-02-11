from __future__ import print_function
print("importing libraries...")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from mpl_toolkits.mplot3d import Axes3D

multiply = 2
frame_length = 2048*multiply*8
hop_length = 1024*multiply
size = 11

print("loading a song...")
filename = "./songs/animals.mp3"
y, sr = librosa.load(filename, sr=22050)

def get_similarity_matrix():
	print("computing similarity matrix...")
	#mfcc = librosa.feature.mfcc(y=y)
	#chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
	chroma = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)
	R = librosa.segment.recurrence_matrix(chroma, mode="distance")
	return R  


def get_checkerboard():
	print("computing checkerboard...")
	row = size
	col = size
	checkerboard = np.fromfunction(lambda i, j: np.sign(i-(row-1)/2)*np.sign(j-(col-1)/2), (row, col), dtype=int)
	tapered = np.fromfunction(lambda i, j: np.exp(-(1*1) * (abs(i-(row-1)/2)**2 + abs(i-(col-1)/2)**2)), (row, col), dtype=int) # obtained from the textbook p209
	return normalize(tapered*checkerboard) # return gaussian tapered checkerboard

def normalize(mat):
	return mat/np.absolute(mat).sum()

def main():
	R = get_similarity_matrix()
	plt.figure(figsize=(10,8))
	librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=hop_length, cmap='gray_r')
	plt.colorbar()

	checkerboard = get_checkerboard()

	print("computing gaussian...")
	arr = []
	for i in range(R.shape[0]-size):
		arr.append((R[i:i+size,i:i+size]*checkerboard).sum()) # value for each point on the diagonal 

	print("complete!")
	plt.figure(figsize=(16, 8))
	arr_time = np.arange(len(arr))*hop_length/22050 # convert from frames to secs
	plt.plot(arr_time, arr)
	plt.show()
	
main()