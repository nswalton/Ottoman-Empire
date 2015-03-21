### Authors: Neil Walton and Stephen Strozyk ###
###              March 2015                  ###

import csv
import sys
import random

def loadData(filename):
    "Load data from the CSV training and test files"
    data = []
    fin = open(filename, 'r')
    line = fin.readline()
    while (line != ''):
        data.append(line.strip('\n'))
        line = fin.readline()
    fin.close()
    return data[1:]


def read_csv(filename):
    """Reads in a csv file and returns a table as a list of lists"""
    the_file = open(filename)
    the_reader = csv.reader(the_file, dialect='excel')
    table = []
    for row in the_reader:
        if len(row) > 0:
            table.append(row)
    return table


def print_csv(table):
    """Prints a csv file from a table as a list of lists (rows)"""
    for row in table:
        for i in range(len(row) - 1):
            sys.stdout.write(str(row[i]) + ',')
        print row[-1]


def getFeatureNames(filename):
    "Return a list of feature names"
    fin = open(filename, 'r')
    line = fin.readline()
    line.strip('\n')
    featureNames = line.split(',')
    return featureNames[1:-1]


def getLabels(data):
    "Returns a list of actual training data labels"
    labels = []
    i = 0
    for datum in data:
        datumRow = datum.split(',')
        if len(datumRow) == 95:
            labels.append(datumRow[94])
    return labels


def extractFeatures(featureNames, datum):
    "Returns a dictionary of feature, value pairs from a training datum"
    features = {}
    idx = 1
    datum = datum.split(',')
    for feat in featureNames:
        features[feat] = datum[idx]
        idx += 1
    return features


def getDataSet(data, featureNames, labels):
    "Returns a list of (feature vector, label) tuples"
    dataSet = []
    idx = 0
    for datum in data:
        if idx < len(labels):
            dataSet.append((extractFeatures(featureNames, datum), labels[idx])) 
        idx += 1
    return dataSet 


def getTestData(data, featureNames):
    dataSet = []
    idx = 0
    for datum in data:
        dataSet.append((extractFeatures(featureNames, datum), "N/A"))
        idx+=1
    return dataSet 


def length_of_table(table):
    """Returns the count of instances in a table"""
    return len(table)


def count_dups(table):
    """Counts duplicates in a given table"""
    count = 0
    for row in table:
        if table.count(row) > 1:
            count+=1
    return count

    
def print_count_csv_dups(table):
    """Prints a csv with a count at the beginning of duplicate rows"""
    for row in table:
        c = table.count(row)
        if c > 1:
            print "count:" + str(c) + "*******************************"
        for i in range(len(row) - 1):
            sys.stdout.write(str(row[i]) + ',')
        print row[-1]


def print_csv_summary(filename):
    """Prints the length and count of duplicates in a csv file"""
    line = "--------------------------------------------------"
    table = read_csv(filename)
    
    print line
    print filename + ":"
    print line

    print "No. of instances:" + str(length_of_table(table))
    print "Duplicates: " + str(count_dups(table))

    
def twod_list(w, h):
    """ Creates a two dimentional list with specified width and height"""
    default = "NA"
    twod_list = []
    new = []
    
    for i in range (0, w):
        for j in range (0, h):
            new.append(default)
        twod_list.append(new)
        new = []
    return twod_list


def oned_list(l):
    """ Creates a one dimentional list with specified length"""
    oned_list = []
    new = []
    
    for i in range (0, l):
        oned_list.append(i)
        new = []
    return oned_list


def wait():
    "Waits for user input"
    print "\n"
    raw_input("Press Enter to continue...")


def holdout_partition(table):
    "Creates Holdout Partitions"
    randomized = table[:]
    n = len(table)    
    for i in range(n):
        j = random.randint(0, n-1)
        randomized[i], randomized[j] = randomized[j], randomized[i]
    n0 = (n*2)/3
    return randomized[0:n0],randomized[n0:]


def randomDraw(number,dataTable):
    ''' draws n number of instances from data.txt at random '''
    i = 0
    rand_Instances = []
    data = read_csv(dataTable)
    max_Val = len(data)
    while(number != i):
        rand_Instances.append(data[random.randint(0, max_Val -1 )])
        i = i +1
    return rand_Instances

    
def kfold_partition(table,k):
    "Creates k roughly equal parts"
    randomized = table[:]
    n = len(table)
    folds = []
    extra = n % k
    foldsize = n / k
    i = 0
    x = 0
    while i < k:
        temp = []
        x = 0;
        while x < foldsize:
            temp.append(randomized[i + (x*k)])
            x = x+1
        if extra > 0:
            temp.append(randomized[i + (x*k)])
            extra = extra-1
        folds.append(temp)
        i = i+1
    return folds

    
def getknn(k, table, columns, columnToPredict,weighted,filename):
    instances = randomDraw(k, str(filename))
    avgs = []
    results = []
    DataTable = table[:]
    predictedClass = []
    for i in range(k):
        NNs = getNearestNeighbors(k, instances[i], columns, DataTable,weighted)
        avgs.append(NNs)
        for NNs in avgs:
            for NN in NNs:
                tempclass = []
                tempclass.append(NN[columnToPredict])
                if tempclass.count("false") >= tempclass.count("true"):
                    predictedClass.append("false")
                else:
                    predictedClass.append("true")
    j = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for instance in instances:
        actual_Class =  instance[columnToPredict]
        predicted_Class = predictedClass[j]
        j += 1
        results.append(actual_Class)
        results.append(predicted_Class)
        if predicted_Class == actual_Class:
            if predicted_Class == "True":
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if predicted_Class == "True":
                FP = FP +1
            else:
                FN = FN +1 
    count = len(results)/2
    accuracy =(TP+TN)/(count * 1.0)
    return accuracy


def majorityVote(predictions,attr):
    votes = []
    for a in attr:
            votes.append(0)
    for prediction in predictions:
        if prediction != -1:
            index=0
            for a in attr:
                if a == prediction:
                    votes[index] =  votes[index] +1
                index = index + 1                
    index = predictions.index(max(predictions))
    return(attr.pop(index))

    
def getNearestNeighbors(k, instance, columns, dataTable,weighted):
    nearest = []
    orderedList = orderByNearestNeighbor(instance, columns, dataTable,weighted)
    i = 1          
    while i <= k:
        nearest.append(orderedList[i])
        i += 1
    return nearest
    

def orderByNearestNeighbor(instance, columns, dataTable,weighted):
    distances = []
    for row in dataTable:
        if weighted != True:
            distances.append(getDistance(instance, row, columns,dataTable))
        else:
            distances.append(getWeightedDistance(instance, row, columns,dataTable))
    return [x for (y,x) in sorted(zip(distances,dataTable))]


def quickBullshitFix(row):
    """remove later"""
    print "this does nothing"

        
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

    
def getDistance(x1, x2, columns,table):
    if x1 == x2:
        return 0.0
    else:
        print x1
        print x2
        sumSqr = 0.0
        for i in columns:
            if is_number(str(x1[i])) == False:
                print x1[i]
                if x1[i] != x2[i]:
                    sumSqr += 1.0
            else:
                totalVals = []
                for row in table:
                   totalVals.append(row[i])
                sumSqr += ((float(x1[i]) - float(x2[i])) ** 2)/(float(max.totalVals-min.totalVals))     
    return sumSqr


def regularKNN(repeats,k, table, columns, columnToPredict,weighted,filename):
    """ Repeats KNN a number of times and its average accuracy avg"""    
    a=0
    a = a+ getknn(k, table, columns, columnToPredict,weighted,filename)
    return( a/(repeats*1.0))


def crossProduct(dict1, dict2):
    """Returns the value of the cross product of two dictionaries"""
    total = 0
    if len(dict1) > len(dict2):
        dict1,dict2 = dict2,dict1
    for key in dict1:
        if key not in dict2:
            continue
        total += dict1[key] * dict2[key]
    return total


def dictAdd(dict1, dict2 ,i):
        """Goes through each key and sums the values for that key from two dictionaries"""
        newDict = {}
        for key in dict1:
            if key in dict2:
                newDict[key] = dict1[key] + dict2[key]
            else:
                newDict[key] = dict1[key]
        for key in dict2:
            if key in dict1:
                continue
            newDict[key] = dict2[key]
        return newDict


def dictSubtract(dict1, dict2, i):
        """Goes through the key and subtracts the values for that key from two dictionaries"""
        newDict = {}
        for key in dict1:
            if key in dict2:
                newDict[key] = dict1[key] - dict2[key]
            else:
                addend[key] = dict1[key]
        for key in dict2:
            if key in dict1:
                continue
            newDict[key] = -1 * dict2[key]
        return newDict


def dataToInts(testDataSet):
    for i in range(len(testDataSet)):
        data = testDataSet[i][0]
        for key in data:
            data[key] = int(data[key])


def argMax(dictionary):
    """Returns the key from a dictionary with the highest value"""
    if len(dictionary.keys()) == 0: return None
    items = dictionary.items()
    values = [x[1] for x in items]
    maxIndex = values.index(max(values))
    return items[maxIndex][0]


def trainPerceptron(validLabels, iterations, trainingDataSet, featureNames):
    """Use the training data to get set values for the training weight vectors"""
    #Set weights to zero initially
    weights = {}
    for label in validLabels:
        weights[label] = {}
        for feat in featureNames:
            weights[label][feat] = 0

    #Train for the specified number of iterations and update weight vector
    trainingData = trainingDataSet
    for iteration in range(iterations):
        print "Starting iteration ", iteration, "..."
        random.shuffle(trainingData)
        for i in range(len(trainingData)):
            true_label = trainingData[i][1]
            data = trainingData[i][0]
            score = {}
            for l in validLabels:
                score[l] = crossProduct(weights[l], data)
            maxguess = argMax(score)
            if maxguess != true_label:
                weights[true_label] = dictAdd(weights[true_label], data, i)
                weights[maxguess] = dictSubtract(weights[maxguess], data, i)
    return weights

    
def classifyPerceptron(data, weights, validLabels):
    """classifies the data as the label whose weight vector is most similar to the data's vector"""
    print "Classifying data using the perceptron..."
    guesses = []
    for datum in data:
        vectors = {}
        for l in validLabels:
            vectors[l] = crossProduct(weights[l], datum[0])
        guesses.append(argMax(vectors))
    return guesses


def trainingStats(trainingGuesses, trainingDataSet):
    i = 0
    correct = 0.0
    for datum in trainingDataSet:
        if datum[1] == trainingGuesses[i]:
            correct += 1.0
        i += 1
    print "Number training correct: ", correct, "       Total: " , 61878
    print "Percentage of training correct:" , str((correct/61878) *100)+"%"


def writeCSV(guesses):
    """Writes a CSV output file containing the formatted guesses for the test labels"""
    print "Writing CSV file..."
    fout = open("classified.csv", 'w')
    fout.write("id, Class_1, Class_2, Class_3, Class_4, Class_5, Class_6, Class_7, Class_8, Class_9")
    for i in range(len(guesses)):
        if guesses[i] == "Class_1":
            writeString = "\n"+str(i+1)+", 1, 0, 0, 0, 0, 0, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_2":
            writeString = "\n"+str(i+1)+", 0, 1, 0, 0, 0, 0, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_3":
            writeString = "\n"+str(i+1)+", 0, 0, 1, 0, 0, 0, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_4":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 1, 0, 0, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_5":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 1, 0, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_6":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 1, 0, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_7":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 1, 0, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_8":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 0, 1, 0"
            fout.write(writeString)
        elif guesses[i] == "Class_9":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 0, 0, 1"
            fout.write(writeString)
    fout.close()
        

def classifierOptionsMenu():
    '''Shows options to users'''
    print "\nClassifier Options"
    print "------------------"
    print "Knn, perceptron\n" # add options as we finish features


def runSelectedClassifier(classifier,trainingDataSet,trainingfile, testDataSet):
    """ Runs selcted calssifier"""
    featureNames = getFeatureNames("train.csv")
    validLabels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    if classifier.lower() == "knn":
        #TODO present options to user        
        print(regularKNN(5,5,trainingDataSet,oned_list(93),94,"false",trainingfile))
    elif classifier.lower() == "perceptron" or classifier.lower() == 'p':
        iterations = int(raw_input("How many iterations should be run?  "))
        trainingWeights = trainPerceptron(validLabels, iterations, trainingDataSet, featureNames)
        trainingGuesses = classifyPerceptron(trainingDataSet, trainingWeights, validLabels)
        trainingStats(trainingGuesses, trainingDataSet)
        testGuesses = classifyPerceptron(testDataSet, trainingWeights, validLabels)
        writeCSV(testGuesses)

        
def main():
    "Select classifier to use and classify the data"
    trainingfile = "train.csv"
    testingfile = "test.csv"
    featureNames = getFeatureNames("train.csv")
    classifierOptionsMenu()
    classifier = raw_input("Which classifier would you like to use?  ")
    print "Preprocessing data..."
    rawTrainingData = loadData(trainingfile)
    rawTestData = loadData(testingfile)
    actualTrainingLabels = getLabels(rawTrainingData)
    trainingDataSet = getDataSet(rawTrainingData, featureNames, actualTrainingLabels)
    testDataSet = getTestData(rawTestData, featureNames)
    dataToInts(testDataSet)
    dataToInts(trainingDataSet)
    runSelectedClassifier(classifier,trainingDataSet,trainingfile, testDataSet)
    
    
    
    


            

if __name__ == '__main__':
    main()
