'''
segment_annotation_and_list_of_words.csv
words_per_segment.json
'''
import numpy as np
from collections import Counter
import json

labels = np.genfromtxt("../Data/segment_annotation_and_list_of_words.csv",delimiter=",", skip_header=True, usecols=[3])

data = json.load(open("../Data/words_per_segment.json","r"))

