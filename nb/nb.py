from math import log
import random
import time

data = 'agaricus-lepiota.data'
features = 'agaricus-lepiota.names'

attribute_pos = []
attribute_neg = []

posdata= [] #----->edible dataset
negdata= [] #----->poison dataset

train_pos = []
train_neg = []

train = []
test = []

mushroom_features = []
features_dictionary = {}

#########################################################
#splitting the dataset into training and testing dataset
#########################################################


with open(data, 'r+') as dataset_file:
        dataset_lines = dataset_file.readlines()

for line in dataset_lines:
    attributes = line.split(',')

    # Get rid of newline character on last attribute

    attributes[-1] = attributes[-1].strip()

    if attributes[0] == 'e':
        posdata.append((attributes[0], attributes[1:]))
    else:
        negdata.append((attributes[0], attributes[1:]))

temp = posdata + negdata
train = temp[:4000]
test = temp[4000:]
print("The size of the training dataset:",(len(train)))
print("The size of the testing dataset:",(len(test)))

#########################################################
#creating a separate list for attributes
#########################################################

def divide_features():
    with open(features, 'r+') as attributes_file:
        for line in attributes_file:
            pair = line.strip().split()
            mushroom_features.append(pair[0])
            features_dictionary[pair[0]] = pair[1].split(',')

def features_data():
    attr_count = 0
    val_count = 0

    for i in range(len(mushroom_features)):
        attribute_pos.append([])
        attribute_neg.append([])

    for i in attribute_pos:
        for j in range(12):
            i.append(0)

    for i in attribute_neg:
        for j in range(12):
            i.append(0)

    for attr in mushroom_features:
        val_count = 0
        for value in features_dictionary[attr]:
            for example in train:
                if value == example[1][attr_count] and example[0] == 'e':
                    attribute_pos[attr_count][val_count] += 1
            val_count += 1
        attr_count += 1
    attr_count = 0

    for attr in mushroom_features:
        val_count = 0
        for value in features_dictionary[attr]:
            for example in train:
                if value == example[1][attr_count] and example[0] == 'p':
                    attribute_neg[attr_count][val_count] += 1
            val_count += 1
        attr_count += 1



#########################################################
#performing the classifier
#########################################################


def naive_bayes(data, neg, pos):
    count = 0
    pos_prob = 1.0
    neg_prob = 1.0

    for i in data:
        pos_prob *= attribute_pos[count][features_dictionary[mushroom_features[count]].index(i)]
        neg_prob *= attribute_neg[count][features_dictionary[mushroom_features[count]].index(i)]
        count += 1

    if neg_prob > pos_prob:
        return 'p'
    else:
        return 'e'

if __name__ == '__main__':
    divide_features()
    features_data()

    num_pos = 0
    num_neg = 0

    for i in range(len(train)):
        if train[i][0] == 'e':
            num_pos += 1
            train_pos.append(train[i][1])
        else:
            num_neg += 1
            train_neg.append(train[i][1])

    correctness = 0

#########################################################
# testing the classifier on testing dataset
#########################################################

    for ex in range(len(test)):
        actual_mushroom = test[ex][0]
        calculated_mushroom = naive_bayes(test[ex][1], num_neg, num_pos)
        # <------>uncomment the bottom line to manually check the accuracy of the code
        #print('actual: %s  classified: %s' % (actual,calculated))
        if actual_mushroom == calculated_mushroom:
            correctness += 1

#########################################################
# displaying the results
#########################################################

    print('\nAccuracy: %f' % (float(correctness*100)/float(len(test))))
