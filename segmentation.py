from __future__ import print_function
print("importing libraries...")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp

multiply = 2
frame_length = 2048*multiply*8
hop_length = 2048*multiply
# kernel size = 32 beats = 32/128 minute = 15 secs = 22050 * 15 samples = 22050 * 15 / 4096 hops = 80 hops
size = 81 # needs to be odd
bpm = 128

print("loading a song...")
filename = "./songs/animals.mp3"
y, sr = librosa.load(filename, sr=22050)

def get_similarity_matrix():
	print("computing similarity matrix...")
	#mfcc = librosa.feature.mfcc(y=y)
	chroma = librosa.feature.chroma_stft(y=y, sr=sr, norm=0, n_fft=frame_length, hop_length=hop_length)
	#chroma = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)
	R = librosa.segment.recurrence_matrix(chroma, mode="distance")
	return R  


def get_checkerboard():
	print("computing checkerboard...")
	row = size
	col = size
	checkerboard = np.fromfunction(lambda i, j: np.sign(i-(row-1)/2)*np.sign(j-(col-1)/2), (row, col), dtype=int)
	print(checkerboard)
	tapered = np.fromfunction(lambda i, j: np.exp(-(0.01*0.01) * ( (i-(row-1)/2)**2 + (j-(col-1)/2)**2) ), (row, col), dtype=int) # obtained from the textbook p209
	print (tapered)
	divisor = np.absolute(checkerboard).sum()
	print("divisor:", (np.absolute(checkerboard).sum()))
	# tapering is in 1D now, so make it 2D
	return tapered*checkerboard # return gaussian tapered checkerboard

def normalize(mat):
	print("divisor:", (np.absolute(mat).sum()))
	return mat/(np.absolute(mat).sum())

def zero_padding(R):
	# could use library as well
	mat = np.zeros((R.shape[0]+size-1,R.shape[1]+size-1))
	mat[(size-1)/2:R.shape[0]+(size-1)/2, (size-1)/2:R.shape[1]+(size-1)/2] = R
	return mat

def main():

	plt.figure(figsize=(6,8))
	gs = gridspec.GridSpec(2, 1,
                       height_ratios=[3, 1]
                       )

	ax1 = plt.subplot(gs[0])

	R = get_similarity_matrix()

	print (R)

	padded_R = zero_padding(R)

	librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=hop_length, cmap='gray_r')

	checkerboard = get_checkerboard()

	print("computing gaussian...")
	arr = []
	for i in range(0, R.shape[0], 1): # hop
		arr.append(abs((padded_R[i:i+size,i:i+size]*checkerboard).sum())) # value for each point on the diagonal 

	print("complete!")
	ax2 = plt.subplot(gs[1])
	arr_time = np.arange(len(arr))*hop_length/22050 # convert from frames to secs
	plt.plot(arr_time, arr)
	plt.xlim(0, len(arr)*hop_length/22050)
	
	plt.plot(arr_time, sp.signal.medfilt(arr, 15))
	plt.show()
main()

# np.set_printoptions(precision=2)
# row=size
# col=size
# checkerboard = np.fromfunction(lambda i, j: np.sign(i-(row-1)/2)*np.sign(j-(col-1)/2), (row, col), dtype=int)
# tapered = np.fromfunction(lambda i, j: np.exp(-(1) * (abs(i-(row-1)/2)**2 + abs(i-(col-1)/2)**2)), (row, col), dtype=int) # obtained from the textbook p209
# print(checkerboard)
# print(tapered)

# check song dependent and bpm dependent
