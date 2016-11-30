'''
segment_annotation_and_list_of_words.csv
words_per_segment.json
'''
import numpy as np
from collections import Counter
import json
import string

def word_cloud(text, name="word_cloud"):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
#     plt.show()
    plt.savefig(name)
    plt.close()

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

video_labels = np.genfromtxt("../Data/segment_annotation_and_list_of_words.csv",delimiter=",", skip_header=True, usecols=[0,3], dtype="|S5")

data = json.load(open("../Data/words_per_segment.json","r"))


data_arr = []
data_arr_0 = []
data_arr_1 = []
data_arr_2 = []

for idx, d in enumerate(data):
    data_arr.append(" ".join(d))
    if video_labels[idx][1] == '0':
        data_arr_0.append(" ".join(d))
    if video_labels[idx][1] == '1':
        data_arr_1.append(" ".join(d))
    if video_labels[idx][1] == '2':
        data_arr_2.append(" ".join(d))
    
    

print len(video_labels)
print len(data_arr)
print len(data_arr_0)
print len(data_arr_1)
print len(data_arr_2)

# word_cloud(" ".join(data_arr),"word_cloud" )
# word_cloud(" ".join(data_arr_0),"word_cloud_0" )
# word_cloud(" ".join(data_arr_1),"word_cloud_1" )
# word_cloud(" ".join(data_arr_2),"word_cloud_2" )


# print Counter(" ".join(data_arr).split(" ")).most_common(500)[-10:]
# print Counter(" ".join(data_arr_0).split(" ")).most_common(500)[-10:]
# print Counter(" ".join(data_arr_1).split(" ")).most_common(500)[-10:]
# print Counter(" ".join(data_arr_2).split(" ")).most_common(500)[-10:]



max_count_feature = []

n_gram_count = Counter([])
n_gram = 3
prev_video = video_labels[0][0]

for idx,d in enumerate(data_arr):
    '''
    Generates n_gram of each segment and finds max_count till now
    '''
    if prev_video != video_labels[idx][0]:
        n_gram_count = Counter([])
    
    # remove puncts
    d = "".join([ i for i in d if i not in string.punctuation ])
    
    # n_gram
    seg_n_gram = find_ngrams(d.split(" "),n_gram)
    
    if not d or len(d) == 0 or d == "" or len(seg_n_gram) == 0:
        max_count_feature.append(0)
        continue
    
    n_gram_count.update(seg_n_gram)
    
    max_gram_count = 0
    for gram in seg_n_gram:
        max_gram_count = max([max_gram_count,n_gram_count[gram]])
    
#     print max_gram_count, seg_n_gram, n_gram_count
    
    max_count_feature.append(max_gram_count)
    

labels = [l[1] for l in video_labels]

# print max_count_feature
# print labels

'''
Trains a boosting tree
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

num_estimators = 5
max_tree_depth = 3

features_tr, features_te, labels_tr, labels_te = train_test_split(np.transpose([max_count_feature]) , labels , train_size = 0.8 , random_state=7)

print len(features_tr), len(features_te), len(labels_tr), len(labels_te) 

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=max_tree_depth), n_estimators=num_estimators)

bdt.fit(features_tr, labels_tr)

err = bdt.estimator_errors_
print "BDT estimator_errors_ max {0}, min {1}, avg {2}, std {3}".format(max(err), min(err), np.average(err), np.std(err))

print("Classification report on sample/test data:")
y_true, y_pred = labels_te, bdt.predict(features_te)
print classification_report(y_true, y_pred)
print "base_est, max_tree_depth, num_estimators, learning_rate",str(base_est) , max_tree_depth, num_estimators, learning_rate


