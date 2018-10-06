import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup 
from nltk.tag import pos_tag

def preprocess(corpus, target_word):
	#  Remove punctuation and digits. Make all words in small letter 
	letters = re.sub("[^a-zA-Z]", " ", corpus)
	small_letters_only = letters.lower()
	#print(small_letters_only)

	#  Remove stopswords
	words = small_letters_only.split()
	no_stopword_data = []
	for w in words:
		if w not in stopwords.words("english"):
			no_stopword_data.append(w)
	#print(no_stopword_data)

	#   Extract surrounding word (-5 to +5 window)
	####################  Problems:   can not identify two target words. Target word should be exect. not like lines, lining etc.
	index = no_stopword_data.index(target_word)
	max_index_of_list = len(no_stopword_data) - 1
	index_low = index - 5
	index_max = index + 5
	if (index_low < 0):
		index_low = 0
	if (index_max > max_index_of_list):
		index_max = max_index_of_list
	no_stopword_data = no_stopword_data[index_low:(index_max + 1)]
	no_stopword_data.remove(target_word)

	lemmatizer = WordNetLemmatizer()
	lemmatized_data = []

	for i in no_stopword_data:
		lemmatized_data.append(lemmatizer.lemmatize(i))
	#print(lemmatized_data)

	return(" ".join(lemmatized_data))

def tag_remover(text_path, target_word, label):
	with open(text_path, 'r') as line_data:
		lines = line_data.read().splitlines()

		clean_data_list = []
		target_Y = []

		for line in lines:
			try:
				word_dic = {}
				final_words = []
				remove_word_ind = []
				words = line.split()
				#print(words)
				#print('#########################################')
				tag_ind = words.index("<tag")
				tag_ind = tag_ind + 1
				h = words[tag_ind]
				d = h.split('>')
				sense = str(d[0])
				#print(sense)
				words[tag_ind] = target_word
				#print(words)
				count = 0
				for word in words:
					if '<' in word:
						remove_word_ind.append(count)
					elif '>' in word:
						remove_word_ind.append(count)
					word_dic[count] = word
					count = count + 1
				#print(remove_word_ind)
				for j in remove_word_ind:
					del word_dic[j]
				for k in word_dic:
					final_words.append(word_dic[k])
				#print(final_words)
				without_tag = " ".join(final_words)
				s = preprocess(without_tag, target_word)
				clean_data_list.append(s)
				target_Y.append(label[sense])			
			except:
				pass

	if (len(clean_data_list) != len(target_Y)):
		print(" Error in data collection")
	return clean_data_list, target_Y



