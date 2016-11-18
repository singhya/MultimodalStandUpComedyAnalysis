import statistics
import fnmatch
import math
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

for openface_file in os.listdir('NormalizedOpenFaceOutput/.'):
	openface_input_file=open('NormalizedOpenFaceOutput/'+openface_file , 'r')
	inp=(openface_input_file.read().split("\n"))[0]
	print openface_file," ",len(inp.split(","))

# a=list()
# a.append(1)
# a.append(2)
# a.append(3)

# print statistics.stdev(a)

# b=list()
# b.append(1)
# b.append(15)
# b.append(19)

# c=list()
# c.append(20)
# c.append(200)
# c.append(0.2)
# print (stats.f_oneway(a,b,c))[1]


# data = [a,b,c]
# labels = ['neutral','positive','negetive']
# plt.ylabel('Gaze(up down) (min_max_diff)')

# plt.boxplot(data,labels= labels)
# plt.show()

# data = [a,b,c]
# labels = ['neutral','positive','negetive']
# plt.ylabel('Gaze(up down) (min_max_diff)')
# plt.boxplot(data,labels= labels)
# plt.show()

# data = [a,b,c]
# labels = ['neutral','positive','negetive']
# plt.ylabel('Gaze(up down) (min_max_diff)')
# plt.boxplot(data,labels= labels)
# plt.show()
