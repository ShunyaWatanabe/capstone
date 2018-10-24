# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time


filename = "./animals.mp3"

start = time.time()
y, sr = librosa.load(filename, sr=22050)
end = time.time()
print("y, sr")
print(y)
print(sr)
print("it took " + str(end-start))

# chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# bounds = librosa.segment.agglomerative(chroma, 8)

# bound_times = librosa.frames_to_time(bounds, sr=sr)

# print(bound_times)

# plt.figure()

# librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')

# plt.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--', linewidth=2, alpha=0.9, label='segmental boundaries')

# plt.axis('tight')

# plt.legend(frameon=True, shadow=True)

# plt.title('Power spectrogram')

# plt.tight_layout()

# plt.show()

mfcc = librosa.feature.mfcc(y=y)

print("mfcc")
print(mfcc)

tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

print("tempo")
print(tempo)

R = librosa.segment.recurrence_matrix(mfcc)

print("R")
print(R)

print("R.shape")
print(R.shape)

length = librosa.get_duration(y=y, sr=sr)
print("length")
print(length)

eight_beats_length = 60 / tempo * 8

length_per_col = length / R.shape[0]

num_cols_per_eight_beats = int(eight_beats_length / length_per_col)

print("num_cols_per_eight_beats")
print(num_cols_per_eight_beats)

counter = float(np.count_nonzero(R))
print("count_nonzero")
print(counter)

global_average = float(counter / (R.shape[0]*R.shape[1]))
print("global_average")
print(global_average)

def calculate_concentration(matrix, start, end):
	count = np.count_nonzero(matrix[start:end, start:end])
	num = float((end-start)*(end-start))
	print("count", count, "num", num, "concentration", float(count/num))
	return float(count/num)

i = 1
while i < int(R.shape[0]/num_cols_per_eight_beats):
	start = int((i-1)*num_cols_per_eight_beats)
	end = int(i*num_cols_per_eight_beats)
	print("start: ", start, "end: ", end)
	trial = calculate_concentration(R, start, end)
	while (trial > global_average):
		second_trial = calculate_concentration(R, start, end+num_cols_per_eight_beats)
		# not the best way to compare, but it's fine for now
		if (trial > second_trial*1.2):
			#print("section changed", "start", str(start), "end", str(end))
			break
		i += 1
		end += num_cols_per_eight_beats
	i += 1
	#print("trial " + str(i) + ": " + str(trial) + str(global_average < trial))


	# the idea is to look at the concentration of trues in the section
	# if increasing, still in the same segment
	# if decreasing, previous stop was the 
	# but also check if the concentration is "high enough", meaning it is a section with enough similarity



# plt.figure()

# librosa.display.specshow(R, x_axis='time', y_axis='time')

# plt.title('Binary recurrence (symmetric)')

# plt.show()