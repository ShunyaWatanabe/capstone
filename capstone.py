# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

print("capstone.py")
filename = "./songs/animals.mp3"
y, sr = librosa.load(filename, sr=22050)

def get_similarity_matrix():
	mfcc = librosa.feature.mfcc(y=y)
	R = librosa.segment.recurrence_matrix(mfcc)
	return R

def get_window(R):
	duration = librosa.get_duration(y=y, sr=sr)
	duration_per_column = duration / R.shape[0]
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
	window_duration = (60 / tempo) * 16 # 16 beats
	window = int(window_duration / duration_per_column)
	return window

def get_average(R):
	sum_trues = float(np.count_nonzero(R))
	average = float(sum_trues / (R.shape[0]*R.shape[1]))
	return average

def get_concentration(matrix, r_start, r_end, c_start, c_end):
	count = np.count_nonzero(matrix[r_start:r_end, c_start:c_end])
	num = float((r_end-r_start)*(c_end-c_start))
	return float(count/num)

def get_segments(R, window, average):
	segments = []
	r_start = 0 # TODO start from where the first count starts
	c_start = 0
	r_end = window
	c_end = window
	segments.append(r_start)
	while r_end <= R.shape[0]:
		conc = get_concentration(R, r_start, r_end, c_start, c_end)
		if conc < average:
			# it's a new section! 
			segments.append(r_start)
			r_start = c_start # this assumes a section is a square
			r_end = c_end
			c_start += window
			c_end += window
			continue
		c_start += window
		c_end += window
	return segments

def display(R):
	plt.figure(figsize=(4, 4))
	librosa.display.specshow(R, x_axis='time', y_axis='time')
	plt.title('Similarity Matrix')
	plt.tight_layout()
	plt.show()

def main():
	print("get_similarity_matrix()")
	R = get_similarity_matrix()
	print(R)
	
	print("get_window(R)")
	window = get_window(R)
	print(window)
	
	print("get_average(R)")
	average = get_average(R)
	print(average)

	print("get_segments(R, window, average)")
	segments = get_segments(R, window, average)
	print (segments)

	length = librosa.get_duration(y=y, sr=sr)
	length_per_col = length / R.shape[0]
	for segment in segments:
		print("new segment at", segment*length_per_col)

	print("display")
	#display(R)
	plt.figure(figsize=(8, 8))
	librosa.display.specshow(R, x_axis='time', y_axis='time')
	plt.title('Similarity Matrix')
	plt.show()

main()