#-*-coding:utf-8-*-

import os

RESULT_VERSION = '01'

DIMENSION = 384
ALPHA = 1.0

INPUT_DIR = os.path.join(os.getcwd(), 'input')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

items = list()
featureRange = list()
featureMean = list()
theta = [0 for i in range(DIMENSION + 1)]
caseNum = 0

def load_from_file(filename):
    global caseNum, theta, items
    with open(os.path.join(INPUT_DIR, filename), 'r') as f:
        for line in f.readlines()[1:]:
            item = list()
            data = line.strip().split(',')
            for num in data[1:]:
                item.append((float) (num))
            items.append(item)
    caseNum = len(items)

def caculate_hypothesis(features):
    hypothesis = 0.0
    for i in range(DIMENSION + 1):
        hypothesis += features[i] * theta[i]
    return hypothesis

def feature_scaling_init():
    global featureRange, featureMean

    featureMin = items[0][:-1]
    featureMax = items[0][:-1]
    featureSum = [0 for i in range(DIMENSION)]
    for item in items:
        item = item[:-1]
        for i in range(len(item)):
            temp = item[i]
            if temp > featureMax[i]: featureMax[i] = temp
            if temp < featureMin[i]: featureMin[i] = temp
            featureSum[i] += temp
    featureRange = [featureMax[i] - featureMin[i] for i in range(DIMENSION)]
    featureMean = [featureSum[i] / caseNum for i in range(DIMENSION)]

def division(divisor, divident):
    if divident == 0:
        return 0.0
    else:
        return divisor / divident

def feature_scaling(features):
    return [division((features[i] - featureMean[i]), featureRange[i]) for i in range(DIMENSION)]

def cost_function():
    cost = 0.0
    for item in items:
        features = feature_scaling(item[:DIMENSION])
        features.append(1.0)
        cost += (caculate_hypothesis(features) - item[DIMENSION]) ** 2
    cost = cost / (2.0 * caseNum)
    return cost

def descent():
    global theta
    for j in range(len(theta)):
        acc = 0.0
        for i in range(caseNum):
            item = items[i]
            features = feature_scaling(item[:DIMENSION])
            features.append(1.0)
            acc += (caculate_hypothesis(features) - item[DIMENSION]) * features[j]

        theta[j] = theta[j] - (ALPHA / caseNum) * acc

def gradient_descent():
    lastCost = 0.0
    unchangeCount = 0

    while unchangeCount < 3:
        # ALPHA: learning rate
        descent()

        cost = cost_function()
        print str(theta), cost
        print '\n'

        diff = lastCost - cost
        if diff > 0 and diff < 0.001:
            unchangeCount += 1
        if diff > 0.001:
            unchangeCount = 0

        lastCost = cost

def gradient_descent_for_test():
    while True:
        # ALPHA: learning rate
        descent()
        print cost_function()

def gen_result_for_test_cases():
    resultFout = open(os.path.join(OUTPUT_DIR, 'result-' + RESULT_VERSION + '.csv'), 'ab')
    resultFout.write('Id,reference\n')

    # load features list frome test samples
    with open(os.path.join(INPUT_DIR, 'test_temp.csv'), 'r') as fin:
        datas = fin.readlines()[1:]

        DIMENSION = len(theta) - 1
        for data in datas:
            data = data.split(',')
            features = feature_scaling([(float) (data[i]) for i in range(1, DIMENSION + 1)])

            estimate = theta[DIMENSION]
            for i in range(DIMENSION):
                estimate += theta[i] * features[i]

            resultFout.write(data[0] + ',' + str(round(estimate, 6)) + '\n')

    resultFout.close()

    logFout = open(os.path.join(OUTPUT_DIR, 'log.txt'), 'ab')
    logFout.write('version: ' + RESULT_VERSION + '\ntheta: ')
    logFout.write(str(theta) + '\n\n')
    logFout.close()

def main():
    # tain
    print 'Loading train set...'
    load_from_file('train_temp.csv')

    print 'Prepare for feature scaling...'
    feature_scaling_init()

    print 'Executing gradient descent...'
    gradient_descent()

    print 'Generating result for test cases...'
    gen_result_for_test_cases()

    print 'Success!'

if __name__ == "__main__":
    main()