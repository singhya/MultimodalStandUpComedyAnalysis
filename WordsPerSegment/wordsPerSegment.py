# Unreasonably high complexity.
import json

f1 = open("segment_annotation.csv")
f2 = open("segment_list_of_words.csv","w")
f2.write("video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,number_of_words,list_of_words_separated_by_spaces\n")

filePath = "C:/Users/rahul/Desktop/CSCI535 Project Dataset/watson_test/"

segmentData = f1.read().split("\n")
words,segNo=[],0

for i in segmentData[1:]:
	data = i.split(",")
	video = data[0]
	segmentStart = data[4]
	segmentEnd = data[5]

	wordFilePath = filePath + video + "_Audio_wordList.csv"
	numberWords=0
	z=0 # dummy variable

	wordData= open(wordFilePath).read().split("\n")
	words.append([])
	for word in wordData:
		try:
			x = word.split(",")
			wordStartTime,wordEndTime,w= float(x[3]),float(x[4]),str(x[2])
			if (float(wordStartTime) >= float(segmentStart) and float(wordEndTime) <= float(segmentEnd)):
				numberWords=numberWords+1
				words[segNo].append(w)
			if wordEndTime > segmentEnd: break
		except:
			z=1

	
	wordString=""
	for wd in words[segNo]:
		wordString= wordString + wd
	segNo=segNo+1
	output = str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "," + str(data[7]) + "," + str(numberWords) + "," +  wordString + "\n"
	f2.write(output)
	print output

f3=open("words_per_segment.json","w")
json.dump(words, f3)
f3.close()
f2.close()	
