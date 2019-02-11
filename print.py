# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

frame_length = [2048, 4096, 8192, 16384]
hop_length = [1024, 2048, 4096, 8192]

print("print.py")
filename = "./songs/animals.mp3"
y, sr = librosa.load(filename, sr=22050)

def get_similarity_matrix(i):
	#mfcc = librosa.feature.mfcc(y=y)
	chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length[i], hop_length=hop_length[1])
	#chroma = librosa.feature.rmse(y=y)
	#chroma = librosa.feature.rmse(y=y, frame_length=frame_length[i], hop_length=hop_length[i])
	R = librosa.segment.recurrence_matrix(chroma)
	return R

def main():
	R_array = []

	for i in range(4):
		R = get_similarity_matrix(i)	
		R_array.append(R)


	plt.figure(figsize=(8, 8))
	for i in range(4):
		plt.subplot(2, 2, i+1)
		librosa.display.specshow(R_array[i], x_axis='time', y_axis='time', hop_length=hop_length[1])
		plt.title('Window Size = ' + str(frame_length[i]))
	plt.suptitle("Similarity Matrix Using Stft with Different Window Sizes")
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()

main()