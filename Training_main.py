from Data_collect import *
from Utils import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import datetime as dt 
import pandas as pd
import pickle

target_word = 'line'
text_path = '/home/saral/NLP/Training/Dataset/line/try.txt'
Result_path = '/home/saral/NLP/Training/Result/'
label = {'"cord"': 0, '"division"': 1, '"phone"': 2, '"product"': 3, '"text"': 4, '"formation"': 5}
dimension = 2  # For SVD (It should be less than total number of rows)


# SVM arguments
kernel_type = 'rbf'
C_value = [0.00097,0.0078125,0.03125, 0.25, 0.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # [2^-10,2^-7, 2^-5, 2^-2, 2^-1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cross_validation = 2
decision_function_shape_name = 'ovo'

#############################################   file path   ##########################################################

vocabulary_csv_name = target_word + '_vocabulary.csv'
vectorizer_name = target_word + '_vectorizer.pkl'
output_weight_name = target_word + '_weight.pkl'
analysis_file_name = target_word + '_analysis.txt'

vocabulary_path = Result_path + vocabulary_csv_name
vectorizer_path = Result_path + vectorizer_name
output_weight = Result_path + output_weight_name
analysis_file_path = Result_path + analysis_file_name



##########################################  Feature extraction  ###########################################################

processed_data, target_Y = tag_remover(text_path, target_word, label)

#print(processed_data)
#print(target_Y)

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 20000)

data_feature = vectorizer.fit_transform(processed_data)
#print(data_feature)
vocab = vectorizer.vocabulary_

s = pd.Series(vocab)
s.to_csv(vocabulary_path, sep=',')

pickle.dump(vocab,open(vectorizer_path,'wb'))

data_feature = data_feature.toarray()
print(data_feature)
#print(data_feature)

data_X = np.ndarray.tolist(data_feature)
print('Number of total samples : {}'.format(len(data_X)))

X_train, X_test, Y_train, Y_test = train_test_split(data_X, target_Y, test_size = 0.3, random_state=42)

print('Number of training samples : {}'.format(len(X_train)))
print('Number of testing samples : {}'.format(len(Y_train)))


###########################################  Singular Value Decomposition (SVD)   #########################################

reconstructed_matrix = SVD(X_train, dimension)
print(reconstructed_matrix)

X_train_svd = np.squeeze(np.asarray(reconstructed_matrix))

X_train_svd = np.ndarray.tolist(X_train_svd)

#print(reconstructed_matrix)

###########################################   Grid search    ###########################################################

param_grid = {'C' : C_value}

classifier = GridSearchCV(svm.SVC(decision_function_shape=decision_function_shape_name, kernel=kernel_type), param_grid, cv=cross_validation)

start_time = dt.datetime.now()
print('start parameter searching at {}'.format(str(start_time)))

classifier.fit(X_train_svd, Y_train)

elapsed_time = dt.datetime.now() - start_time
print('Elapsed time : {}'.format(str(elapsed_time)))

#print(classifier.cv_results_['mean_test_score'])

print('Best Estimator is : {}'.format(classifier.best_estimator_))

print('Best score : {}'.format(classifier.best_score_))

print('Best parameter : {}'.format(classifier.best_params_))

joblib.dump(classifier, output_weight)

print('Model has been saved')

f = open(analysis_file_path, 'w')
f.write('###############################   Training Results   #################################' + '\n')
f.write('\n' + 'Elapsed Time:' + '\n' + str(elapsed_time) + '\n')
bst_esti = classifier.best_estimator_
f.write('\n' + 'Best Estimator:' + '\n' + str(bst_esti) + '\n')
bst_score = classifier.best_score_
f.write('\n' + 'Best Score:' + '\n' + str(bst_score) + '\n')
bst_par = classifier.best_params_
f.write('\n' + 'Best parameter:' + '\n' + str(bst_par) + '\n')



########################################   Evalution of model    #####################################################

MODEL = joblib.load(output_weight)

expected = Y_test

predicted = MODEL.predict(X_test)

print('Predicted labels: {}'.format(predicted))

report = classification_report(expected, predicted)

print('Classification report : {}'.format(report))

Confusion_mat = confusion_matrix(expected, predicted)

print('Confusion matrix: \n {}'.format(Confusion_mat)) 

accuracy = accuracy_score(expected, predicted)

print('Accuracy : {}'.format(accuracy))

f.write('\n' +'##########################      Evalution Results     #################################' + '\n')
f.write('\n' + 'Report:' + '\n' + str(report) + '\n')
f.write('\n' + 'Confusion Matrix' + '\n' + str(Confusion_mat) + '\n')
f.write('\n' + 'Accuracy: ' + '\n' + str(accuracy))
f.close()