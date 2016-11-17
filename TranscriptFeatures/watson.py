import json
from pprint import pprint
import os

ELONGATION_COEFFICIENT=1.1
count=0
filePath = "C:/Users/rahul/Desktop/CSCI535 Project Dataset/watson_final"
for fileName in os.listdir(filePath):

	outputFileName1= fileName.split(".")[0] + "_wordList.csv"
	outputFileName2= fileName.split(".")[0] + "_phraseList.csv"

	f1=open(outputFileName1 , "w")
	f2=open(outputFileName2 , "w")

	with open(os.path.join(filePath, fileName)) as fp:    
	    data = json.load(fp)

	cnt=0
	index=[]
	words=[]
	uniqueWords={}
	phrases=[]
	for element in data:
		results = element["results"]
		for res in results:
			if res["final"] == True:
				cnt=cnt+1
				wordList=res["alternatives"][0]["timestamps"]
				for word in wordList:
					if word[0] not in uniqueWords.keys():
						uniqueWords[word[0]]= [word[2] - word[1],1]
					else:
						uniqueWords[word[0]]= map(sum, zip(uniqueWords[word[0]],[word[2] - word[1],1]))
					words.append(fileName +"," +str(element["result_index"]) +","+ word[0]+","+str(word[1])+","+str(word[2])+","+str(word[2] - word[1]))
				try:
					phrases.append(fileName +"," +str(element["result_index"]) +","+res["alternatives"][0]["transcript"] + ","+ str(wordList[0][1]) + "," + str(wordList[-1][2]) + "\n")
				except:
					print "Error in Reading transcripts for index", element["result_index"]
				index.append(element["result_index"])
				#print "\n"



	for k,v in uniqueWords.items():
		v.append(ELONGATION_COEFFICIENT*(float(v[0])/v[1]))
		#print k,v

	words2=[]
	for word in words:
		word=word.split(",")
		#print word[5], uniqueWords[word[2]][2] 
		if float(word[5]) > float(uniqueWords[word[2]][2]):
			word.append("1")
		else:
			word.append("0")
		#print str(word[0]) + "," +str(word[1]) +"," + str(word[2]) + "," +str(word[3]) + "," +str(word[4]) + "," +str(word[5]) + "," +str(word[6]) + "\n"
		words2.append(str(word[0]) + "," +str(word[1]) + "," +str(word[2]) + "," +str(word[3]) + "," +str(word[4]) + "," +str(word[5]) + "," +str(word[6]) + "\n")

	count=count +1
	print "Done " + fileName + str(count)
	for word in words2:
		f1.write(word)
	for phrase in phrases:
		f2.write(phrase)
	f1.close()
	f2.close()	
