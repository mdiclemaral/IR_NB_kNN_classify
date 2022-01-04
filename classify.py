
"""
Created by Maral Dicle Maral, January 2022
Information Retrieval - Text Classification (Naive Bayes vs KNN Classification)
Works with: python3 classify.py folder_directory stop_words_directory
"""
from bs4 import BeautifulSoup
import sys

import os
import time
from multiprocessing import Process, Manager
import math
import string
from numpy.linalg import norm
import numpy as np
from random import seed
from random import randint

start_time = time.time()



"""
Reads an individual file and the stop_words file, tokenizes, cleans stop words and punctuations of the docs. 
Changes result to dictionary of : {id:[topic class, train or test status, tokens of the doc]}. 
        doc_list_for_inverted_idx = {id = [token1,token2..]}
        topics

"""
def processor(dir, stops, input, result, topics, doc_list_for_inverted_idx):

    stop_words = open(stops).read().split()
    punct = string.punctuation

    file_handler = open(dir + input, 'r', encoding='latin1').read()
    soup = BeautifulSoup(file_handler, 'html.parser')

    reuters = soup.find_all('reuters')

    #Finds the top 10 topics

    for r in reuters:


        if r['topics'] == 'YES' and (not r.body == None or not r.title == None):
            topic_text = str(r.topics)
            topic_text = topic_text.replace('<topics>', '')
            topic_text = topic_text.replace('</topics>', '')
            topic_text = topic_text.replace('</d>', '')
            all_t = topic_text.split('<d>')
            all_t = all_t[1:]
            if len(all_t) == 0:
                continue
            for m in all_t:
                if m in topics:
                    topics[m] += 1
                else:
                    topics[m] = 1


    #Reads reuters documents, tokenizes, cleans stop words and punctuations

    for r in reuters:
        if r['topics'] == 'YES' and (not r.body == None or not r.title == None):
            train_status= r['lewissplit']
            id = int(r['newid'])
            topic_text= str(r.topics)
            topic_text= topic_text.replace('<topics>','')
            topic_text = topic_text.replace('</topics>', '')
            topic_text= topic_text.replace('</d>','')
            topic_for_result = topic_text.split('<d>')
            topic_for_result = topic_for_result[1:]
            if len(topic_for_result) == 0:
                continue
            text = r.text.lower()

            for t in text:
                if t in punct:
                    text = text.replace(t, " ")

            text = text.replace('\n', ' ')
            tokens = text.replace('\t', ' ')

            tokenized = tokens.split()
            tokenized1 = []
            for t in tokenized:
                if not t in stop_words:
                    tokenized1.append(t)
            result[id] = [topic_for_result, train_status, tokenized1]
            doc_list_for_inverted_idx[id]= tokenized1

"""
Computes the document frequencies for the words in the created inverted index file, enters the 
idf weights of the words into idf_index

Returns idf_index
"""
def compute_idf(index, num_words_in_docs):

    total_docs = max(list(num_words_in_docs.keys()))
    idf_idx = {}
    for word in index:
        idf_idx[word] = math.log((total_docs/len(index[word])), 10)

    return idf_idx

'''
Creates the idf values of words based on the created inverted index.
#inverted_idx= {'mohawk': {8000: [16, 22, 33, 112]}, 'nmk': {8000: [23]}, 'petitions': {8000: [156]}}

Returns idf_vals_docs= {w1:idf1, w2:idf2}
        num_of_words_in_docs  {doc1:num1, doc2:num2}
'''
def inverted_idxer(docs):

    num_words_in_docs = {}
    inverted_idx = {}
    for key in docs:
        ID = int(key)
        words = docs[key]
        num_words_in_docs[ID] = len(words)
        for num_word in range(0, len(words)):
            word = words[num_word]

            if not word in inverted_idx:
                position_dict = {}
                position_dict[ID] = [num_word]
                inverted_idx[word] = position_dict
            else:
                if ID in inverted_idx[word]: # If docID is already added to the inverted index, continue
                    inverted_idx[word][ID].append(num_word)
                else:
                    inverted_idx[word][ID] = [num_word]
    idf_vals_docs = compute_idf(inverted_idx, num_words_in_docs)

    return inverted_idx, idf_vals_docs, num_words_in_docs

'''
Generates test and training groups from the extracted word dictionary of reuters dataset. 

Returns  train_set= {class1={doc1=[word1,word2..],doc2=[w1,w2..]},class2={..}}, 
         test_set 
         y_real= {doc1:[class1,class2..], doc2=[c1,c2]} 
'''
def train_test_generate(w_dict, classification_topics):

    test_set = {}
    y_real = {}
    train_set = {}
     # result[id]= [topic class, train or test status, tokens of the doc]
    for id, value in w_dict.items():

        topics_interested = list(set(value[0]) & set(classification_topics))
        if value[1]=='TEST' and len(topics_interested) != 0:
            y_real[id] = topics_interested
        for topic in topics_interested:
            if value[1] == 'TEST':
                if topic not in test_set:
                    test_set[topic]={}
                test_set[topic][id]= value[2]

            elif value[1] == 'TRAIN':
                if topic not in train_set:
                    train_set[topic] = {}
                train_set[topic][id] = value[2]

    return test_set, train_set, y_real

'''
Trains the multi-label naive bayes classifier. Computes P(cj), P(wk | cj) calculation, 
and it computes total number of words (from set structure) in the training set for normalization

Returns  P_wk_cj = {class1: {word1: P(wk | cj)}, class2: {}}
         P_cj = {class1: P(cj1), class2: P(cj1)}
         
         # For normalization: 
         num_all_words_for_normalization = num of type of of words in the whole training set. (how many times a word occurs is NOT considered)
         class_num_words = num of words count in the classes  (how many times a word occurs is considered)
        
'''
def train_naive_bayes(sample_set):
    all_token_set = {'9'}
    P_cj = {}
    class_num_words = {}
    bag_for_classes = {}
    alpha = 1
    total_docs_num = 0
    #Computes the total number of docs in the training set
    for nb_class, class_content in sample_set.items():
        total_docs_num += len(class_content)

    #Computes P(cj), keeps the bag of words for P(wk | cj) calculation
    #Computes total number of words (from set structure) in the training set for normalization
    for nb_class, class_content in sample_set.items():
        P_cj[nb_class]= len(class_content) / total_docs_num
        num_of_words_for_class = 0
        bag_of_words = {}
        for id, tokens in class_content.items():
            all_token_set.update(tokens)
            for token in tokens:
                if not token in bag_of_words:
                    bag_of_words[token] = 1
                else:
                    bag_of_words[token] += 1
                num_of_words_for_class += 1
        bag_for_classes[nb_class] = bag_of_words
        class_num_words[nb_class] = num_of_words_for_class
    num_all_words_for_normalization = len(all_token_set)


    #Computes and keeps P(wk | cj) for each word in each class
    for nb_class, bag_of_words in bag_for_classes.items():
        for word, value in bag_of_words.items():
            normalized_probabilty = (value + alpha) / (class_num_words[nb_class] + (alpha * num_all_words_for_normalization))
            bag_of_words[word] = normalized_probabilty
        bag_for_classes[nb_class] = bag_of_words

    P_wk_cj = bag_for_classes

    return P_cj, P_wk_cj, num_all_words_for_normalization, class_num_words


'''
Predicts the topic classes of the test set based on trained naive bayes model. 
Predicts only the best fitting class. The multi-class option is not considered.

Returns y_pred = {doc_id: 'pred_topic, doc_id2: 'pred_topic2'}

'''
def predict_naive_bayes(P_cj, P_wk_cj, num_all_words_for_normalization, class_num_words, test_set):
    result = {}
    for nb_class, docs in test_set.items():
        for id, tokens in docs.items():
            result_topic = {}
            for topic in class_num_words.keys():
                p_c = math.log(P_cj[topic])
                p_x_c = 0
                for token in tokens:
                    if not token in P_wk_cj[topic]:
                        p_token = math.log(1/(class_num_words[topic] + num_all_words_for_normalization))
                    else:
                        p_token = math.log(P_wk_cj[topic][token])
                    p_x_c += p_token
                p_doc_for_class = p_c + p_x_c
                result_topic[topic] = p_doc_for_class
            result[id]=result_topic
    y_pred={}
    for id, class_val in result.items():
        sorted_val = sorted(class_val.items(), key=lambda a: a[1], reverse=True)
        first_class= sorted_val[0][0]
        y_pred[id] = first_class

    return y_pred

'''
Vectorizes the y_pred, y_real values of the test set according to the given classification topics
(if y_pred includes class1, from the set of topics [class1, class2, class3], its vectorized form is [1,0,0])

Returns vectorized_y_real = [[1001..0],.., [001..1]]
        vectorized_y_pred 

'''
def y_vectorizer(y_dict, classification_topics):
    vectorized_y_real = []
    vectorized_y_pred = []
    n_topic = len(classification_topics)
    for id, y_values in y_dict.items():
        list_of_y_real = [0] * n_topic
        list_of_y_pred = [0] * n_topic
        y_pred= y_values[0]
        y_real= y_values[1]
        idx_pred = classification_topics.index(y_pred)
        list_of_y_pred[idx_pred] = 1
        for y in y_real:
            idx_real = classification_topics.index(y)
            list_of_y_real[idx_real] = 1
        vectorized_y_pred.append(list_of_y_pred)
        vectorized_y_real.append(list_of_y_real)

    return vectorized_y_real, vectorized_y_pred

'''
Finds the accuracy measures for given vectorized y_real and y_pred lists:
Returns f1_macro
        macro_precision
        macro_recall
        micro_precision
        micro_recall
'''
def accuracy_meter(vectorized_y_real, vectorized_y_pred, classification_topics):

    class_vals= {}
    for sample_i in range(len(classification_topics)):
        true_positive = 0.0001
        true_negative = 0.0001
        false_positive = 0.0001
        false_negative = 0.0001
        for i in range(len(vectorized_y_pred)):
            if vectorized_y_real[i][sample_i] == 1 and vectorized_y_pred[i][sample_i] == 1:
                true_positive += 1
            elif vectorized_y_real[i][sample_i] == 0 and vectorized_y_pred[i][sample_i] == 1:
                false_positive += 1
            elif vectorized_y_real[i][sample_i] == 1 and vectorized_y_pred[i][sample_i] == 0:
                false_negative += 1
            elif vectorized_y_real[i][sample_i] == 0 and vectorized_y_pred[i][sample_i] == 0:
                true_negative += 1
            else:
                print('Input is incorrect for accuracy calculations!')
                break
        class_vals[classification_topics[sample_i]]=[true_positive, true_negative, false_positive, false_negative]

    class_accuracies = {}
    macro_precision, macro_recall = 0, 0
    tp_total, fp_total, fn_total = 0, 0, 0

    for class_name, c_v in class_vals.items(): #{'earn': [1053, 1246, 4, 30], 'acq': [690, 1608, 29, 6]}
        tp = c_v[0]
        tn = c_v[1]
        fp = c_v[2]
        fn = c_v[3]

        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = tp / (fp + tp)
        recall = tp / (fn + tp)

        f1 = 2 * (precision * recall) / (precision + recall)

        macro_precision += precision
        macro_recall += recall

        class_accuracies[class_name] = [precision, recall, f1]
        #print(class_name, 'presicion: ', precision,'  recall: ', recall, ' f1: ', f1)
    micro_recall = tp_total / (tp_total + fp_total)
    micro_precision = tp_total / (tp_total + fn_total)
    macro_recall = (macro_recall / len(classification_topics))
    macro_precision = (macro_precision / len(classification_topics))
    f1_macro = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)


    return f1_macro, macro_precision, macro_recall, micro_precision, micro_recall

'''
Runs the randomization test to find the p value for 2 given classification results, 
in order to find if their classification skills are significantly different or is it because of coinsidence

Returns p value
'''
def randomization_test(y_dict_1, y_dict_2, classification_topics):

    v_y_real_1, v_y_pred_1 = y_vectorizer(y_dict_1, classification_topics)
    v_y_real_2, v_y_pred_2 = y_vectorizer(y_dict_2, classification_topics)

    f1_1, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(v_y_real_1, v_y_pred_1, classification_topics)
    f1_2, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(v_y_real_2, v_y_pred_2, classification_topics)
    s = abs(f1_1 - f1_2)
    count = 0
    R = 10
    seed(1)
    for k in range(R):
        for i in range(len(v_y_pred_2)):
            rand_val = randint(0, 1)
            if rand_val == 1:
                v_y_pred_2[i],  v_y_pred_1[i] = (v_y_pred_1[i],  v_y_pred_2[i]) #????
        n_f1_1, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(v_y_real_1, v_y_pred_1, classification_topics)
        n_f1_2, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(v_y_real_2, v_y_pred_2, classification_topics)
        n_s = abs(n_f1_1 - n_f1_2)
        if n_s >= s:
            count += 1

    p_value = (count + 1) / (R + 1)
    print('P value', round(p_value,4))
    return p_value


'''
Selects the best 1 percent features according to their term frequency in the training set. 

Returns [feature1, feature2..]
'''
def feature_select_w_tf(X):
    #X = {class1 = {doc1 = [word1, word2..], doc2 = [w1, w2..]}, class2 = {..}}
    freq_words={}
    for class_name, docs_in_class in X.items():
        for doc_name, words in docs_in_class.items():
            seen_words = []
            for word in words:
                if word in freq_words:
                    if word not in seen_words:
                        freq_words[word] += 1
                        seen_words.append(word)
                else:
                    freq_words[word] = 1
    sorted_freq_words = sorted(freq_words.items(), key=lambda a: a[1], reverse=True)
    percent_1 = int(len(sorted_freq_words) * 0.01)

    selected_features = []
    for s in range(percent_1):
        selected_features.append(sorted_freq_words[s][0])

    return selected_features

'''
Finds the tf-idf vectors by calculating the tf-idf values for each word (feature) in the document. 
Uses only the given features for vectorization.
Returns tf_idf_vectors_train = [feature1, feature2...]
        tf_idf_vectors_train_names = [[id, topic],[id2,topic2]]
        tf_idf_vectors_test 
        tf_idf_vectors_test_names
'''
def tf_idf_vectorizer(features, idf_vals_docs, w_dict, classification_topics):

    # w_dict[id]= [topic class, train or test status, tokens of the doc]
    tf_idf_vectors_test = []
    tf_idf_vectors_test_names = []
    tf_idf_vectors_train_names = []
    tf_idf_vectors_train = []
    for id, doc in w_dict.items():
        topics_interested = list(set(doc[0]) & set(classification_topics))
        if len(topics_interested) != 0:
            topic = topics_interested
            test_stat = doc[1]
            words_doc = doc[2]
            vector_for_doc = []
            for feature in features:
                tf = (words_doc.count(feature) / len(words_doc))
                if tf == 0:
                    tf_idf = 0
                else:
                    tf_idf = (1 + math.log(tf, 10)) * idf_vals_docs[feature]
                vector_for_doc.append(tf_idf)
            if test_stat == 'TRAIN':
                tf_idf_vectors_train_names.append([id, topic])
                tf_idf_vectors_train.append(vector_for_doc)
            elif test_stat == 'TEST':
                tf_idf_vectors_test_names.append([id, topic])
                tf_idf_vectors_test.append(vector_for_doc)

    '''
    ids = []
    for i in tf_idf_vectors_train_names:
        ids.append(i[0])
    print('test set boyut', len(ids))
    #print(ids)
    norm_dump = open("ids.pkl", "wb")
    pickle.dump(ids, norm_dump)

    multi_train=0
    class_count = {}
    for c in classification_topics:
        class_count[c]= 0
    for t in tf_idf_vectors_train_names: #teste çevir test countları verir
        if len(t[1]) >1:
            multi_train+= 1
        for i in t[1]:
            class_count[i]+=1

    #print(tf_idf_vectors_test_names)
    print('trains with multi label', multi_train)
    print('counts of the labels', class_count)
    plt.bar(class_count.keys(), class_count.values())
    '''

    return tf_idf_vectors_train, tf_idf_vectors_train_names, tf_idf_vectors_test, tf_idf_vectors_test_names


"""
Computes the cosine similarity of a test vector and the whole training set. 

Returns cos_sim vector [cos_sim_for_train1, cos_sim_for_train2..]
"""
def cosine_similarity1(vec1, vec2_list):

    vec1 = np.array(vec1, dtype=np.float)
    vec2_list = np.array(vec2_list, dtype=np.float)
    cos_sim = vec2_list.dot(vec1) / (np.linalg.norm(vec2_list, axis=1) * np.linalg.norm(vec1))

    return cos_sim

"""
Runs the knn algorithm by calling the find_knn_neighbor() 
(finds the best fitting y_pred for the test) on the tf-idf vectorized test set. 
"""
def knn_multi_process(k, tf_idf_vectors_test, tf_idf_vectors_test_names, tf_idf_vectors_train, tf_idf_vectors_train_names, classification_topics, y_dict):
    for i in range(len(tf_idf_vectors_test)):
        test_vec = tf_idf_vectors_test[i]
        id = tf_idf_vectors_test_names[i][0]
        y_real = tf_idf_vectors_test_names[i][1]
        y_pred = find_knn_neighbor(k, tf_idf_vectors_train, tf_idf_vectors_train_names, test_vec, classification_topics)
        y_dict[id] = [y_pred, y_real]

"""
KNN classification algorithm. Runs the knn classification algorithm on the test set.
Divides the test set into 4 and runs the algorithm on parallel in 4 groups by calling knn_multi_process()
Returns y_dict= {doc_id= [y_pred, [y_real1,y_real2] ]}
"""
def knn(k, tf_idf_vectors_train, tf_idf_vectors_train_names, tf_idf_vectors_test, tf_idf_vectors_test_names, classification_topics):
    #w_dict[id] = [topic_for_result, train_status, tokenized1]

    per= int(0.25 * len(tf_idf_vectors_test))
    a= tf_idf_vectors_test[:per]
    a_n= tf_idf_vectors_test_names[:per]
    b= tf_idf_vectors_test[per:2*per]
    b_n= tf_idf_vectors_test_names[per:2 * per]
    c= tf_idf_vectors_test[2*per:3*per]
    c_n= tf_idf_vectors_test_names[2*per:3 * per]
    d= tf_idf_vectors_test[3*per:]
    d_n= tf_idf_vectors_test_names[3*per:]
    list_of_test=[[a,a_n],[b,b_n],[c,c_n],[d,d_n]]

    manager = Manager()
    y_dict = manager.dict()
    processes=[]
    for l in list_of_test:
        process = Process(target=knn_multi_process, args=(k, l[0], l[1], tf_idf_vectors_train,
                      tf_idf_vectors_train_names, classification_topics, y_dict))
        processes.append(process)
        process.start()
    for i in range(len(processes)):
        processes[i].join()
    return y_dict

'''
Finds the closest k neighbors of a given test vector in terms of their (cosine similarity). 
Finds the best fitting topic in terms of the best k neighbors' topic class. 

Returns the best fitting topic class to the test vector as a str. (ie 'earn')
'''
def find_knn_neighbor(k, tf_idf_vectors_train, tf_idf_vectors_train_names, test_vec, classification_topics):
    topic_count= {}
    for t in classification_topics:
        topic_count[t]= 0

    cos_sim = cosine_similarity1(test_vec, tf_idf_vectors_train)
    idx_cos_sim= np.argpartition(cos_sim, -k)[-k:]

    for idx in idx_cos_sim:
        topic = tf_idf_vectors_train_names[idx][1]
        topic_count[topic[0]] += 1
    sorted_topic_count = sorted(topic_count.items(), key=lambda a: a[1], reverse=True)
    y_pred = sorted_topic_count[0][0]

    return y_pred

'''
Returns the best k value for knn classification based on the f1 scores ranging from 7 to 15. 
A subset of the training set was used as the development set.

Returns k integer
'''
def find_best_k(tf_idf_vectors_train, tf_idf_vectors_train_names, classification_topics):
    best_k=0
    best_f1=0
    percent_5 = int(0.05 *len(tf_idf_vectors_train))
    percent_45= int(0.45 *len(tf_idf_vectors_train))

    dev = tf_idf_vectors_train[:percent_5]
    dev_names = tf_idf_vectors_train_names[:percent_5]
    train = tf_idf_vectors_train[percent_5:percent_45]
    train_names = tf_idf_vectors_train_names[percent_5:percent_45]

    for k in range(7,30):
        y_dict_knn = knn(k, train, train_names, dev, dev_names, classification_topics)
        vectorized_y_real_knn, vectorized_y_pred_knn = y_vectorizer(y_dict_knn, classification_topics)
        f1_macro, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(vectorized_y_real_knn, vectorized_y_pred_knn, classification_topics)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_k = k
        else:
            break

    return best_k

"""
Creates a new process for each file in reuters directory calls processor() for each file. 
Naive Bayes and KNN Classification is called.
"""
def fileHandler(dir, stops):
    #File Handling
    files = os.listdir(dir)
    input_starter = 'reut'
    manager = Manager()
    w_dict = manager.dict()
    topics = manager.dict()
    w_dict_for_inverted_idx = manager.dict()
    processes = []
    results = []
    count = 0

    #Preprocess of the data
    for f in files:
        if input_starter in f:
            temp_dict = {}
            process = Process(target=processor, args=(dir, stops, f, w_dict, topics, w_dict_for_inverted_idx))

            results.append(temp_dict)
            processes.append(process)
            process.start()
            count += 1
    for i in range(len(processes)):
        processes[i].join()

    # Finds the first ten most frequent topics which will be used for classification.
    sorted_topics = sorted(topics.items(), key=lambda a: a[1], reverse=True)

    classification_topics = []
    for k in range(10):
        classification_topics.append(sorted_topics[k][0])



    ####### Naive Bayes Classification ######

    #Classification
    test_set, train_set, y_real = train_test_generate(w_dict, classification_topics)


    P_cj, P_wk_cj, num_all_words_for_normalization, class_num_words = train_naive_bayes(train_set)

    y_pred_nb = predict_naive_bayes(P_cj, P_wk_cj, num_all_words_for_normalization, class_num_words, test_set)
    y_dict_nb = {}

    for id, y_val in y_real.items():
        y_dict_nb[id] = [y_pred_nb[id], y_real[id]]

    vectorized_y_real_nb, vectorized_y_pred_nb = y_vectorizer(y_dict_nb, classification_topics)


    #Accuracy calculation
    f1_macro, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(vectorized_y_real_nb, vectorized_y_pred_nb, classification_topics)
    print('#### ACCURACY SCORES FOR NAIVE BAYES ####')
    print('Macro Precision: ', round(macro_precision,4), '   Macro Recall: ', round(macro_recall,4), '  Macro F1 Score: ', round(f1_macro,4))
    print('Micro Precision: ', round(micro_precision,4), '   Micro Recall: ', round(micro_recall,4))
    print('')


    ####### KNN Classification ######
    
    #Feature selection
    selected_features = feature_select_w_tf(train_set) #bunu trainsete baglı yapma vectorizerın içindeki seçmeceyi kullan ve knn ve nbyi ayırmış ol
    inverted_idx, idf_vals_docs, num_words_in_docs = inverted_idxer(w_dict_for_inverted_idx)
    
    #TF_IDF Vectorization
    tf_idf_vectors_train, tf_idf_vectors_train_names, tf_idf_vectors_test, tf_idf_vectors_test_names = tf_idf_vectorizer(selected_features, idf_vals_docs, w_dict, classification_topics)
    
    #Finding best scoring k 
    best_k = find_best_k(tf_idf_vectors_train, tf_idf_vectors_train_names, classification_topics)
    
    #Classification
    y_dict_knn = knn(best_k, tf_idf_vectors_train, tf_idf_vectors_train_names, tf_idf_vectors_test, tf_idf_vectors_test_names, classification_topics)
    vectorized_y_real_knn, vectorized_y_pred_knn = y_vectorizer(y_dict_knn, classification_topics)
    
    #Accuracy calculation
    f1_macro, macro_precision, macro_recall, micro_precision, micro_recall = accuracy_meter(vectorized_y_real_knn, vectorized_y_pred_knn, classification_topics)
    print('#### ACCURACY SCORES FOR KNN ####')
    print('Macro Precision: ', round(macro_precision,4), '   Macro Recall: ', round(macro_recall,4), '  Macro F1 Score: ', round(f1_macro,4))
    print('Micro Precision: ', round(micro_precision,4), '   Micro Recall: ', round(micro_recall,4))
    print('')
    
    #Randomization test
    randomization_test(y_dict_nb, y_dict_knn, classification_topics)


def main():
    stops = sys.argv[2] #'stopwords.txt'
    dir = sys.argv[1] #'./reuters21578/'
    fileHandler(dir, stops)

if __name__ == '__main__':
    main()
    run_time = time.time() - start_time
    print("- Classification process ends in %.6f seconds -" % (round(run_time, 5)))


