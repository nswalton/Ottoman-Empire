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
        datumRow = datum
        if len(datum) == 95:
            labels.append(datum[94])
    return labels


def sumOfFeatures(rawDatum):
    """Returns the sum of all feature values for a instance of datum"""
    idx = 1
    total = 0
    while idx < 94:
        total += rawDatum[idx]
        idx += 1
    return total


def topThreeSum(datum, features):
    """Adds the categorization of the sum of the three maximum feature values to the feature vector"""
    total = 0
    tempDatum = datum
    for i in range(3):
        total += max(tempDatum[1:-1])
        tempDatum.remove(max(tempDatum))
    if total < 30:
        features['low'] = 3
        features['high'] = 0
    else:
        features['low'] = 0
        features['high'] = 3


def totalFeaturesOn(datum, features):
    i = 1
    total = 0
    while i < 94:
        if datum[i] > 0:
            total += 1
        i += 1
    features["num_feats"] = total


def binaryFeats(datum, features):
    i = 1
    while i < 94:
        strNum = str(datum[i])
        featureName = "feat_" + str(i)
        if datum[i] > 0:
            features[featureName] = 1
        else:
            features[featureName] = 0
        i += 1


def stdDevFeats(datum, features):
    means = [0.3866802418, 0.2630660332, 0.9014674036, 0.7790814183, 0.0710430201, 0.0256957238, 0.1937037396, 0.6624325285, 1.011296422, 0.2639063965, 1.2528685478, 0.1408739778, 0.4809787, 1.6966934936, 1.2843983322, 1.4134587414, 0.3661075019, 0.5754226058, 0.5516985035, 0.4715246129, 0.2040143508, 0.729968648, 0.1425223828, 2.6438798927, 1.5345195384, 0.5631080513, 0.6966126895, 0.2389702317, 0.2757684476, 0.1503119041, 0.14867966, 1.0437958564, 0.6965157245, 0.9464106791, 0.6662626459, 0.7090888523, 0.2636316623, 0.5821293513, 0.4855845373, 1.6530592456, 0.3034681147, 0.6980186819, 0.451145803, 0.5608293739, 0.2381298685, 0.6413749636, 0.2496687029, 1.5848928537, 0.3483144252, 0.3242832671, 0.0532984259, 0.2134845987, 0.4420634151, 2.0724651734, 0.3231196871, 0.3037751705, 0.3091082453, 0.6979701994, 0.3886033808, 1.029929862, 0.2397459517, 1.1875626232, 0.1685898057, 1.2567956301, 0.2222276092, 0.5717056143, 2.8976534471, 0.3929021623, 0.8111283493, 0.8927890365, 0.3192895698, 0.8587220014, 0.5910501309, 0.5798506739, 0.7268172856, 0.7484566405, 0.1241959986, 0.3664145577, 0.300446039, 0.6980671644, 0.0784608423, 0.1879828049, 0.496719351, 0.0707521251, 0.5323055044, 1.1285755842, 0.3935485956, 0.8749151556, 0.4577717444, 0.8124212159, 0.2649406897, 0.3801189437, 0.1261352985]

    standardDevs = [1.5253172391, 1.2520632386, 2.9347941478, 2.7879820716, 0.4388988422, 0.215331727, 1.0300933268, 2.2557513489, 3.474794391, 1.0833310973, 3.0423079993, 0.5670841497, 2.0146812069, 3.1631860329, 3.8622051759, 2.2261447561, 1.4774241975, 1.3359740409, 4.6361071026, 1.4387150084, 0.6960444593, 1.4462086781, 0.7829728993, 4.6289772503, 2.3329746493, 1.7102910746, 2.8731983213, 0.8281051991, 1.9012784328, 1.6408663474, 0.8973472144, 2.4168291775, 1.3101914055, 3.3685947207, 3.197939512, 2.5550985997, 0.7569279525, 1.6025656903, 3.2982885192, 3.2997709389, 1.0856637108, 1.9611731739, 1.705999313, 1.3460790944, 2.5871104804, 2.3483403388, 1.4461909219, 2.5770506493, 1.3693685262, 1.7204559041, 0.5138160622, 1.0447799537, 2.0064692124, 4.1132853174, 0.9987349781, 1.9257901419, 1.0821390976, 3.9836894847, 2.5776716765, 3.0284444264, 1.0175451752, 2.6667206479, 0.9461508244, 3.4020523706, 0.7830456845, 1.3618631218, 4.9742815066, 1.761039449, 4.1110581687, 1.9413520475, 1.1624338258, 2.4116267192, 5.7831859212, 3.7577913024, 3.2000691863, 2.920014247, 0.9066139727, 2.7782942983, 1.2855589498, 2.245652536, 0.4612400393, 0.8362618701, 2.4349012725, 1.1514506945, 1.9004224367, 2.6815327467, 1.5754421335, 2.1154493486, 1.5273726512, 4.5977673188, 2.0456290274, 0.9823766451, 1.2017104421]
    i = 1
    while i < 94:
        featureName = "feat__" + str(i)
        if datum[i] == 0:
            features[featureName] = 0
        else:
            value = 2 * abs((datum[i] - means[i-1])/float(standardDevs[i-1]))
            features[featureName] = value
        i += 1
            


def extractFeatures(featureNames, datum, noBasic):
    "Returns a dictionary of feature, value pairs from a training datum"
    features = {}
    idx = 1
    for feat in featureNames:
        if noBasic: break
        features[feat] = datum[idx]
        idx += 1
    #features["sum"] = sumOfFeatures(datum)
    #topThreeSum(datum, features)

    #binaryFeats(datum, features)
    totalFeaturesOn(datum, features)
    stdDevFeats(datum, features)
    #topThreeSum(datum, features)
    return features


def getDataSet(data, featureNames, labels, noBasic):
    "Returns a list of (feature vector, label) tuples"
    dataSet = []
    idx = 0
    for datum in data:
        if idx < len(labels):
            dataSet.append((extractFeatures(featureNames, datum, noBasic), labels[idx])) 
        idx += 1
    return dataSet 


def getTestData(data, featureNames, noBasic):
    dataSet = []
    idx = 0
    for datum in data:
        dataSet.append((extractFeatures(featureNames, datum, noBasic), "N/A"))
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


def isStephenALizardPerson():
    answer = raw_input("\nIs Stephen a lizard person? (y/n)  ")
    if answer.lower() == 'y' or answer.lower() == "yes":
        print "You are correct, Stephen is a lizard person."
    elif answer.lower() == 'n' or answer.lower() == "no":
        print "OH NO, THE LIZARD HAS CORRUPTED YOUR BRAIN. Stephen is in fact the most lizardy lizard person ever to have lived."
        print "Let's see if you learned your lesson, I'll ask you again..."
        isStephenALizardPerson()
    else:
        print "That was a bullshit response. Let's try again:"
        isStephenALizardPerson()

        
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


def dataToInt(rawData):
    for i in range(len(rawData)):
        rawData[i] = rawData[i].split(',')
        j = 0
        while j < 94:
            rawData[i][j] = int(rawData[i][j])
            j += 1


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
    #r = random.random()
    r = 0.814011835135
    for i in range(9): 
        random.shuffle(trainingData, lambda: r)
    
    #random.shuffle(trainingData)
    for iteration in range(iterations):
        print "Starting iteration ", iteration, "..."
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


def classifyWithPerceptron(data, weights, validLabels):
    """Classify with probabilities instead of binary outcome."""
    print "Classifying data using the perceptron...(with probabilities)..."
    guesses = []
    for datum in data:
        vectors = []
        for l in validLabels:
            vectors.append(crossProduct(weights[l], datum[0]))
        guesses.append(vectors)
    return guesses


def writeGuesses(guesses):
    """Takes a list of lists called guesses and writes a csv file with probabilites
       Each list in the list of guesses contains the probability for each label (in order of class 1-9)"""
    print "Writing CSV file with probabilities..."
    fout = open("toSubmit.csv", 'w')
    fout.write("id, Class_1, Class_2, Class_3, Class_4, Class_5, Class_6, Class_7, Class_8, Class_9")
    for i in range(len(guesses)):
        total = 0.0
        for j in range(len(guesses[i])):
            if guesses[i][j] > 0: total += float(guesses[i][j])
        writeLine = '\n' + str(i+1) + ", "
        for k in range(len(guesses[i])):
            if guesses[i][k] > 0:
                writeLine = writeLine + str(guesses[i][k]/float(total)) + ", "
            else:
                writeLine = writeLine + str(0) + ", "
        fout.write(writeLine[:-2])
    fout.close()


def trainingStats(trainingGuesses, trainingDataSet):
    """Print the number/percentage of correct guesses for the training data set"""
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
        elif guesses[i] == "Class_2":
            writeString = "\n"+str(i+1)+", 0, 1, 0, 0, 0, 0, 0, 0, 0"
        elif guesses[i] == "Class_3":
            writeString = "\n"+str(i+1)+", 0, 0, 1, 0, 0, 0, 0, 0, 0"
        elif guesses[i] == "Class_4":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 1, 0, 0, 0, 0, 0"
        elif guesses[i] == "Class_5":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 1, 0, 0, 0, 0"
        elif guesses[i] == "Class_6":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 1, 0, 0, 0"
        elif guesses[i] == "Class_7":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 1, 0, 0"
        elif guesses[i] == "Class_8":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 0, 1, 0"
        elif guesses[i] == "Class_9":
            writeString = "\n"+str(i+1)+", 0, 0, 0, 0, 0, 0, 0, 0, 1"
        fout.write(writeString)
    fout.close()


def findStats(data, features):
    print "Getting statistics about data..."
    validLabels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    stats = {}
    triples = {} #
    for l in validLabels:
        triples[l] = [] #
        stats[l] = {}
        for feat in features:
            stats[l][feat] = 0
        stats[l]["count"] = 0
    for datum in data:
        label = datum[94]
        i = 1
        stats[label]["count"] += 1
        while i < 94:
            keyString = "feat_"+str(i)
            stats[label][keyString] += datum[i]
            i += 1

    for j in range(3):
        for key in stats:
            maxKey = argMax(stats[key])
            #print key, maxKey
            triples[key].append(maxKey)
            del stats[key][maxKey]
    for lab in validLabels:
        total = 0
        for key in triples[lab]:
            i = 1
            intKey = key[5:]
            intKey = int(intKey)
            for datum in data:
                total += datum[intKey]
        print "The average sum of top three features for label   ", lab, "   is:", float(total)/stats[lab]["count"]
                
    
def getDataSetFeatureNames(dataSet):
    feats = []
    d = dataSet[0][0]
    for key in d:
        feats.append(key)
    return feats
        

def classifierOptionsMenu():
    '''Shows options to users'''
    print "\nClassifier Options"
    print "------------------"
    print "Knn, perceptron\n" # add options as we finish features


def runSelectedClassifier(classifier,trainingDataSet,trainingfile, testDataSet):
    """ Runs selcted calssifier"""
    featureNames = getDataSetFeatureNames(trainingDataSet)
    validLabels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    if classifier.lower() == "knn":
        #TODO present options to user        
        print(regularKNN(5,5,trainingDataSet,oned_list(93),94,"false",trainingfile))
    elif classifier.lower() == "perceptron" or classifier.lower() == 'p':
        iterations = int(raw_input("How many iterations should be run?  "))
        trainingWeights = trainPerceptron(validLabels, iterations, trainingDataSet, featureNames)
        trainingGuesses = classifyPerceptron(trainingDataSet, trainingWeights, validLabels)     #using binary outcomes 
        trainingStats(trainingGuesses, trainingDataSet)     #using binary outcomes
        #testGuesses = classifyPerceptron(testDataSet, trainingWeights, validLabels) #binary outcomes
        testGuesses = classifyWithPerceptron(testDataSet, trainingWeights, validLabels)     #for probabilities, not binary outcome
        writeGuesses(testGuesses)       #write probabilities, not binary
        #writeCSV(testGuesses)  

        
def main():
    "Select classifier to use and classify the data"
    #isStephenALizardPerson()
    trainingfile = "train.csv"
    testingfile = "test.csv"
    #trainingfile = "fakeTrain.csv"
    #testingfile = "fakeTest.csv"
    featureNames = getFeatureNames("train.csv")
    classifierOptionsMenu()
    classifier = raw_input("Which classifier would you like to use?  ")

    engineeredFeatures = raw_input("Would you like to use engineered features only (no raw features)  (y/n)?  ")
    noBasic = False
    if engineeredFeatures.lower() == 'y' or engineeredFeatures.lower() == "yes":
        noBasic = True

    print "Preprocessing data..."
    rawTrainingData = loadData(trainingfile)
    rawTestData = loadData(testingfile)
    dataToInt(rawTrainingData)
    dataToInt(rawTestData)

    #findStats(rawTrainingData, featureNames)    #for printing out some statistics about the data set

    actualTrainingLabels = getLabels(rawTrainingData)
    trainingDataSet = getDataSet(rawTrainingData, featureNames, actualTrainingLabels, noBasic)
    testDataSet = getTestData(rawTestData, featureNames, noBasic)
    runSelectedClassifier(classifier,trainingDataSet,trainingfile, testDataSet)
    
    
    
    


            

if __name__ == '__main__':
    main()
