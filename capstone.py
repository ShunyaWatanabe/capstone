# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

# frame_length = 2048*16
# hop_length = 10335
multiply = 2
frame_length = 2048*multiply*8
hop_length = 1024*multiply

print("capstone.py")
filename = "./songs/animals.mp3"
y, sr = librosa.load(filename, sr=22050)

def get_similarity_matrix():
	#mfcc = librosa.feature.mfcc(y=y)
	#chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
	#chroma = librosa.feature.rmse(y=y)
	chroma = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)
	R = librosa.segment.recurrence_matrix(chroma)
	return R

def get_window(R):
	duration = librosa.get_duration(y=y, sr=sr)
	duration_per_column = duration / R.shape[0]
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
	print("tempo", tempo)
	window_duration = (60 / tempo) * 8
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
	first = True
	while r_end <= R.shape[0]:
		conc = get_concentration(R, r_start, r_end, c_start, c_end)
		if conc < average:
			# need to align the first block, so if the first try gives a false alert, slide the window little by little
			if first:
				r_start += 5
				c_start += 5
				c_start += 5
				c_end += 5
				continue
			# it might be a new section. let's double check
			temp_start = r_start
			temp_end = r_end
			# we will try majority rule this time
			max_counter = min((c_start - r_start)/window, 1)
			counter = 1.0
			while r_start < c_start:
				# we slide the window upwards
				r_start += window
				r_end += window
				conc = get_concentration(R, r_start, r_end, c_start, c_end)
				if conc < average:
					counter += 1.0
			if float(counter/max_counter) > 0.75:
				# new section! update everything
				r_start = c_start
				r_end = c_end
				segments.append(r_start)
			else:
				# false alert. revert back 
				r_start = temp_start
				r_end = temp_end
		first = False
		c_start += window
		c_end += window
	return segments

def display(R, i):
	plt.subplot(2, 2, i)
	librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=hop_length)
	plt.title('hop length = ' + str(hop_length))


def main():
	global multiply, frame_length, hop_length
	R_array = []

	for i in range(1, 5):

		print("get_similarity_matrix()")
		R = get_similarity_matrix()
		print("R.shape[0]",R.shape[0])
		
		'''
		print("get_window(R)")
		window = get_window(R)
		print("window", window)
		
		print("get_average(R)")
		average = get_average(R)

		print("get_segments(R, window, average)")
		segments = get_segments(R, window, average)

		length = librosa.get_duration(y=y, sr=sr)
		length_per_col = length / R.shape[0]
		
		for segment in segments:
			print(segment*length_per_col)
		for segment in segments:
			#print(segment*length_per_col)
			minute = int(segment*length_per_col/60)
			seconds = int(segment*length_per_col%60)
			if (seconds/10 == 0):
				print(str(minute)+ ":" + "0" + str(seconds))
			else:
				print(str(minute)+ ":" + str(seconds))
		'''
		R_array.append(R)
		multiply = multiply*2
		#frame_length = 2048*multiply
		hop_length = 1024*multiply

	plt.figure(figsize=(8, 8))
	for index, r in enumerate(R_array):
		print("display", index+1)
		display(r, index+1)
	print("show")

	#plt.suptitle('Similarity Matrix with different window size')
	plt.tight_layout()
	plt.show()

main()