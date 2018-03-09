import csv
import random
import math
import operator

#########################################################
#loading the dataset
#########################################################

def load_data(filename, split, training=[], test=[]):
    with open(filename, 'rt') as csvfile:
        row = csv.reader(csvfile)
        dataset = list(row)
        for i in range(len(dataset) - 1):
            for j in range(4):
                dataset[i][j] = float(dataset[i][j])
            if random.random() < split:
                training.append(dataset[i])
            else:
                test.append(dataset[j])

#########################################################
#implementing euclidean distance
#########################################################

def euclidean_distance(neighbor, host, length):
    distance = 0
    for i in range(length):
        distance += pow((neighbor[i] - host[i]), 2)
    return math.sqrt(distance)

#########################################################
#knn classifier
#########################################################

def myknnclassify(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for i in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#########################################################
#knn regressor
#########################################################

def myknnregress(train, test, k):
    distances = []
    length = len(test) - 1
    for i in range(len(train)):
        dist = euclidean_distance(test, train[i], length)
        distances.append((train[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


######################################################################
#selecting the maximum value for the neighbor in the classification
######################################################################

def get_classify(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#########################################################
#selecting the mean for the neighbor in the classification
#########################################################

def get_regress(neighbors,u):
    class_list = []
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        class_list.append(response)
        class_list = list(map(int, class_list))
    value = sum(class_list)
    mean = float((value)/(u))
    return str(math.floor(mean))


#########################################################
#calculating accuracy
#########################################################

def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0



def main():
    # prepare dataset and splitting into 60% training dataset and 40% testing dataset
    X = []
    test = []
    split = 0.80

##################################################################################
# comment out the bottom code for the implementation of knn classification
##################################################################################


    # load_data('iris.csv', split, X, test)
    # print('Length of training dataset: ' + repr(len(X)))
    # print('Length of testing dataset: ' + repr(len(test)))
    # predictions = []
    # k_classify = int(input("Enter the value of k (knn classify): "))
    # for x in range(len(test)):
    #     neighbors = myknnclassify(X, test[x], k_classify)
    #     print(neighbors)
    #     result = get_classify(neighbors)
    #     predictions.append(result)
    #     print('Predicted=' + repr(result) + ', Actual=' + repr(test[x][-1]))
    # accuracy = get_accuracy(test, predictions)
    # print('Accuracy: ' + repr(accuracy) + '%')

#-----------------------------------------------------------------------------------------------------------------#

##################################################################################
# comment out the bottom code for the implementation of knn regression
##################################################################################
#
    # load_data('iris.data', split, X, test)
    # print('Length of training dataset: ' + repr(len(X)))
    # print('Length of testing dataset: ' + repr(len(test)))
    # predictions = []
    # k_regress = int(input("\nEnter the value of k (knn regress): "))
    # for x in range(len(test)):
    #     y = myknnregress(X, test[x], k_regress)
    #     print(y)
    #     result = get_regress(y,k_regress)
    #     print(result)
    #     predictions.append(result)
    #     print('Predicted=' + repr(result) + ', Actual=' + repr(test[x][-1]))
    # accuracy = get_accuracy(test, predictions)
    # print('Accuracy: ' + repr(accuracy) + '%')

main()
