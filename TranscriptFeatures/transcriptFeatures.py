# Unreasonably high complexity.

f1 = open("segment_annotation.csv")
f4 = open("segment_annotation_transcript_features.csv","w")
f4.write("video,laughter_start,laughter_end,laughter_value,start_segment_s,end_segment_s,start_segment_frame,end_segment_frame,number_of_words, number_of_elongated_words, number_of_continuous_segments, number_of_pauses, max_pause_time , min_pause_time, avg_pause_time, last_pause_length\n")

filePath = "C:/Users/rahul/Desktop/CSCI535 Project Dataset/watson_test/"

segmentData = f1.read()

segmentData = segmentData.split("\n")

for i in segmentData[1:]:
	data = i.split(",")
	video = data[0]
	segmentStart = data[4]
	segmentEnd = data[5]

	wordFilePath = filePath + video + "_Audio_wordList.csv"
	phraseFilePath = filePath + video + "_Audio_phraseList.csv"
	
	numberPauses=0
	numberContinuousSegments=0
	pauses=[]
	lastPauseLength=0
	numberWords=0
	numberElongatedWords=0
	z=0 # dummy variable

	wordData= open(wordFilePath).read().split("\n")
	phraseData= open(phraseFilePath).read().split("\n")

	for word in wordData:
		try:
			x = word.split(",")
			wordStartTime,wordEndTime= float(x[3]),float(x[4])
			if (float(wordStartTime) >= float(segmentStart) and float(wordEndTime) <= float(segmentEnd)):
				numberWords=numberWords+1
				if int(x[6]) == 1: numberElongatedWords = numberElongatedWords + 1
			if wordEndTime > segmentEnd: break
		except:
			z=1
	
	for j in range(len(phraseData)):
		x = phraseData[j].split(",")
		try:
			xPlus = phraseData[j+1].split(",")
		except:
			xPlus = [0,0,0,0,0,0,0,0,0,0,0]

		try:
			phraseStartTime,phraseEndTime= float(x[3]),float(x[4])
			if (float(phraseStartTime) >= float(segmentStart) and float(phraseEndTime) <= float(segmentEnd)):
				numberContinuousSegments=numberContinuousSegments + 1
				print xPlus[3], x[4]
				if float(xPlus[3]) - float(x[4]) > 0 :
					pauses.append(float(xPlus[3]) - float(x[4]))
					lastPauseLength = float(xPlus[3]) - float(x[4])
				
			if phraseEndTime > segmentEnd: break
		
		except:
			z=1
	
	numberPauses = numberContinuousSegments - 1
	if numberPauses <0 : numberPauses =0
	
	try:
		maxPause = max(pauses)
		minPause = min(pauses)
		avgPause = float(sum(pauses)) / len(pauses)
	except:
		maxPause, minPause, avgPause = 0,0,0

	output = str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + "," + str(data[7]) + "," + str(numberWords) + "," + str(numberElongatedWords) + "," + str(numberContinuousSegments) + "," +  str(numberPauses) + "," + str(maxPause) + "," + str(minPause) + "," + str(avgPause) + "," + str(lastPauseLength) + "\n"
	f4.write(output)
	print output

f4.close()	
