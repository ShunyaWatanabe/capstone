from os import listdir, path
from song import Song
import json

def make_data(song_title, duration, bpm, cue_points, segments):
	data = {}  
	data["song_title"] = song_title
	data["duration"] = duration
	data["bpm"] = bpm
	data["cue points"] = cue_points
	data["segments"] = []
	for segment in segments:
		data["segments"].append({
			"start" : segment.start, 
			"end" : segment.end, 
			"segment_type" : segment.segment_type, 
			"energy": str(segment.energy),
			"key" : "Ab",
			"bpm" : bpm
		})
	return data

def main():
	bpm = 128
	frame_length = 2048*2*8
	hop_length = 2048*2
	song_titles = listdir("database/songs")
	song_titles.remove(".DS_Store")
	song_titles.remove(".gitkeep")

	songs = []
	# analyse songs
	for index, song_title in enumerate(song_titles):
		# if index == 5:
		# 	break
		print(str(index+1) + ": working on " + song_title)
		annotation = "database/annotations/" + song_title.replace(".mp3", ".txt")
		song_title = "database/songs/" + song_title
		songs.append(Song(song_title, annotation, bpm, frame_length, hop_length))


	# get average f1 value
	total = 0
	for song in songs:
		f1 = song.ROC(15, 15)
		if (f1 < 0.5):
			song.plot()
		total += f1
	average = total/len(songs)

	print("average f1 = ", average)

	# make the data
	data = {}
	data["songs"] = []
	data["segments"] = {
		"intro": [],
		"verse": [],
		"build-up": [],
		"drop": [],
		"break": [],
		"outro": [],
	}

	for song in songs:
		datum = make_data(path.basename(song.filename), song.duration, song.bpm, song.times, song.segments)
		data["songs"].append(datum)
		for segment in song.segments:
			if not segment.segment_type in ["intro","verse","build-up","drop","break","outro"]:
				continue
			data["segments"][segment.segment_type].append({
				"song_title" : path.basename(song.filename),
				"start" : segment.start, 
				"end" : segment.end,
				"energy": str(segment.energy),
				"key" : "Ab",
				"bpm" : bpm
			})

	for key, value in data["segments"].items():
		data["segments"][key] = sorted(data["segments"][key], key=lambda k: k["energy"]) 

	# save the data in a file
	with open("database/data/data.txt", "w") as outfile:  
		json.dump(data, outfile)

main()
