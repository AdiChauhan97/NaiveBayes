import pandas as pd
import numpy as np
import random
import math

#cleaning the data set
data = pd.read_csv('spam_filter\SMSSpamCollection', sep='\t', header=None, engine='python', quoting=3)
data[1] = data[1].str.replace('\W+', ' ', regex=True).str.replace('\s+', ' ', regex=True).str.strip()
data[1] = data[1].str.lower()
data[1] = data[1].str.split()
data = data[data[1].astype(bool)]
class_count = data.groupby(0).count()
sort_df = data.sort_values(0)
ham = sort_df.iloc[:class_count[1].iloc[0]]
spam = sort_df.iloc[class_count[1].iloc[0]:]

#created stratified cross validation function for training and testing due to the dataset classes being imbalanced
#this function returns indexes to use on the dataset
def strat_cross_validation(k):
    data_ham = []
    data_spam = []
    index_ham = list(range(0, len(ham)))
    index_spam = list(range(0, len(spam)))
    rand_index_ham = random.sample(index_ham, len(index_ham))
    rand_index_spam = random.sample(index_spam, len(index_spam))
    np_index_ham = np.array(rand_index_ham)
    np_index_spam = np.array(rand_index_spam)
    for split in np.array_split(np_index_ham, k):
        sp = split.tolist()
        data_ham.append(sp)
    for split in np.array_split(np_index_spam, k):
        sp = split.tolist()
        data_spam.append(sp)
    for i in range(k):
        test_index_ham = data_ham[i]
        test_index_spam = data_spam[i]
        train_ham = data_ham.copy()
        train_spam = data_spam.copy()
        train_ham.pop(i)
        train_spam.pop(i)
        train_index_ham = [item for sublist in train_ham for item in sublist]
        train_index_spam = [item for sublist in train_spam for item in sublist]
        yield train_index_ham, train_index_spam, test_index_ham, test_index_spam

#function to count word frequency
def wordfreq(data):
    unique, counts = np.unique(data, return_counts=True)
    freq = dict(zip(unique, counts))
    return freq

def naivebayes(X, y, X_test):
    classes_count = wordfreq(y)
    values = classes_count.values()
    total_classes = sum(values)
    words_ham = []
    words_spam = []
    all_words = []
    all_test_words = []
    cond_prob_ham = {}
    cond_prob_spam = {}
    predict = {}
    prior = {}
    final_prediction = []
    for key, value in classes_count.items():
        prior[key] = value / total_classes    
        
    for count, i in enumerate(y):
        if i == 'ham':
            words_ham.extend(X[count])
        elif i == 'spam':
            words_spam.extend(X[count])
        all_words.extend(X[count])
        
    all_unique_words = wordfreq(all_words)
    words_ham_count = wordfreq(words_ham)
    words_spam_count = wordfreq(words_spam)
    
    for i in X_test:
        all_test_words.extend(i)

    unique_words_test = wordfreq(all_test_words)
    for test_key in unique_words_test.keys():
        value_ham = words_ham_count.get(test_key)
        if value_ham != None:
            cond_prob_ham[test_key] = value_ham + 1 / (len(words_ham) + len(all_unique_words))
        if value_ham == None:
            cond_prob_ham[test_key] = 0 + 1 / (len(words_ham) + len(all_unique_words))
                
        value_spam = words_spam_count.get(test_key)
        if value_spam != None:
            cond_prob_spam[test_key] = value_spam + 1 / (len(words_spam) + len(all_unique_words))
        else:
            cond_prob_spam[test_key] = 0 + 1 / (len(words_spam) + len(all_unique_words))
                
    count = 0
    for abstract in X_test:
        abs_count = wordfreq(abstract)
        for key, value in abs_count.items():
            predict['ham'] = predict.get('ham', 0) + value * math.log(cond_prob_ham.get(key))
            predict['spam'] = predict.get('spam', 0) + value * math.log(cond_prob_spam.get(key))
        final_ham =  predict.get('ham') + math.log(prior.get('ham'))
        final_spam =  predict.get('spam') + math.log(prior.get('spam'))
        final = {'ham': final_ham, 'spam': final_spam}
        letter = max(final, key=final.get)  
        final_prediction.append(letter)
        predict.clear()
    return final_prediction

#score function to compare naive bayes results to test results 
def score(nb, y):
    n = np.array(nb)
    correct = (n == y)
    accuracy = correct.sum() / correct.size
    return accuracy

#applys the indexes from the strat_cross_validation function to the actual dataset then
#passes it into the naive bayes function and runs the score function on the results to see the accuracy of our algorithm
accuracy = []
for i in range(5):
    for train_ham, train_spam, test_ham, test_spam in strat_cross_validation(10):
        X_train_ham, X_test_ham = ham[1].iloc[train_ham], ham[1].iloc[test_ham]
        y_train_ham, y_test_ham = ham[0].iloc[train_ham], ham[0].iloc[test_ham]
        X_train_spam, X_test_spam = spam[1].iloc[train_spam], spam[1].iloc[test_spam]
        y_train_spam, y_test_spam = spam[0].iloc[train_spam], spam[0].iloc[test_spam]
        X_train, X_test = np.array(pd.concat([X_train_ham, X_train_spam])), np.array(pd.concat([X_test_ham, X_test_spam]))
        y_train, y_test = np.array(pd.concat([y_train_ham, y_train_spam])), np.array(pd.concat([y_test_ham, y_test_spam]))
        nb = naivebayes(X_train, y_train, X_test)
        result = score(nb, y_test)
        accuracy.append(result)
    total_acc = np.mean(np.array(accuracy))
    print(f"Naive Bayes accuracy using 10 fold CV: {total_acc:.3f}")
    accuracy.clear()
