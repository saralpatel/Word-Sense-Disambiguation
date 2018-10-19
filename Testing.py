from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import re


model_path = '/home/saral/NLP/Final_project/Testing/Model_inputs/line_weight.pkl'
vectorizer_path = '/home/saral/NLP/Final_project/Testing/Model_inputs/line_vectorizer.pkl'

test_path = '/home/saral/NLP/Final_project/Testing/test.txt'

# In Posiible_target_word, first word must be base word.
possible_target_word = ['line', 'lines']
label = {0: '"cord"', 1: '"division"',2: '"phone"', 3: '"product"', 4: '"text"', 5: '"formation"'}


#########################################   Data preprocessing    ########################################


def preprocess(test_path, target_words):
	feature_list = []
	sentence_number = []
	with open(test_path, 'r') as line_data:
		lines = line_data.read().splitlines()

		sentence_no = 0
		print('Number of lines : {}'.format(len(lines)))

		for line in lines:
			letters = re.sub("[^a-zA-Z]", " ", line)
			small_letters_line = letters.lower()
			words = small_letters_line.split()
			flag = False
			for i in words:
				if i in target_words:
					target_ind = words.index(i)
					flag = True
			if (flag == False):
				for j in words:
					if target_words[0] in j:
						target_ind = words.index(i)
						flag = True
			if (flag == False):
				sentence_no = sentence_no + 1
				continue
			elif(flag == True):
				words[target_ind] = target_words[0]
				sentence_number.append(sentence_no)
			
			no_stopword_data = []
			for w in words:
				if w not in stopwords.words("english"):
					no_stopword_data.append(w)
			index = no_stopword_data.index(target_words[0])
			max_index_of_list = len(no_stopword_data) - 1
			index_low = index - 5
			index_max = index + 5
			if (index_low < 0):
				index_low = 0
			if (index_max > max_index_of_list):
				index_max = max_index_of_list
			no_stopword_data = no_stopword_data[index_low:(index_max + 1)]
			no_stopword_data.remove(target_words[0])

			lemmatizer = WordNetLemmatizer()
			lemmatized_data = []

			for k in no_stopword_data:
				lemmatized_data.append(lemmatizer.lemmatize(k))

			feature = " ".join(lemmatized_data)
			feature_list.append(feature)
			sentence_no = sentence_no + 1
	return feature_list, sentence_number

#########################################   Prediction   ##############################################

processed_data, sentence_number = preprocess(test_path, possible_target_word)
#print(processed_data)

classifier = joblib.load(model_path)

vocab = pickle.load(open(vectorizer_path, 'rb'))

with open(test_path, 'r') as line_data:
	lines = line_data.read().splitlines()

vectorizer = CountVectorizer(analyzer = "word", vocabulary = vocab,tokenizer = None, preprocessor = None, stop_words = None, max_features = 20000)


data = vectorizer.transform(processed_data)
data = data.toarray()
data_X = np.ndarray.tolist(data)
result = classifier.predict(data_X)

ind = 0
for m in result:
	print(lines[sentence_number[ind]] + '\n')
	print(label[m] + '\n')
	ind = ind + 1