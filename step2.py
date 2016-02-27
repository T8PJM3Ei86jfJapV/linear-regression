#-*-coding:utf-8-*-

dimension = 384
caseNum = 0
theta = list()
items = list()
featureRange = list()
featureMean = list()

def load_from_file(filename):
    global caseNum, theta, items
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            item = list()
            data = line.strip().split(',')
            for num in data[1:]:
                item.append((float) (num))
            items.append(item)
    caseNum = len(items)

    # load theta
    with open('origin-theta-for-step2.txt', 'r') as fin:
        content = ''.join(fin.read().split())[1:-1]
        theta = [(float) (feature) for feature in content.split(',')]

def caculate_hypothesis(features):
    hypothesis = 0.0
    for i in range(dimension + 1):
        hypothesis += features[i] * theta[i]
    return hypothesis

def feature_scaling_init():
    global featureRange, featureMean

    featureMin = items[0][:-1]
    featureMax = items[0][:-1]
    featureSum = [0 for i in range(dimension)]
    for item in items:
        item = item[:-1]
        for i in range(len(item)):
            temp = item[i]
            if temp > featureMax[i]: featureMax[i] = temp
            if temp < featureMin[i]: featureMin[i] = temp
            featureSum[i] += temp
    featureRange = [featureMax[i] - featureMin[i] for i in range(dimension)]
    featureMean = [featureSum[i] / caseNum for i in range(dimension)]

def division(divisor, divident):
    if divident == 0:
        return 0.0
    else:
        return divisor / divident

def feature_scaling(features):
    return [division((features[i] - featureMean[i]), featureRange[i]) for i in range(dimension)]

def cost_function():
    cost = 0.0
    for item in items:
        features = feature_scaling(item[:dimension])
        features.append(1.0)
        cost += (caculate_hypothesis(features) - item[dimension]) ** 2
    cost = cost / (2.0 * caseNum)
    return cost

def descent(alpha):
    global theta
    for j in range(len(theta)):
        acc = 0.0
        for i in range(caseNum):
            item = items[i]
            features = feature_scaling(item[:dimension])
            features.append(1.0)
            acc += (caculate_hypothesis(features) - item[dimension]) * features[j]

        theta[j] = theta[j] - (alpha / caseNum) * acc

def gradient_descent(alpha):
    lastCost = 0.0
    unchangeCount = 0

    while unchangeCount < 3:
        # alpha: learning rate
        descent(alpha)

        cost = cost_function()
        print str(theta), cost
        print '\n'

        diff = lastCost - cost
        if diff > 0 and diff < 0.003:
            unchangeCount += 1
        if diff > 0.003:
            unchangeCount = 0

        lastCost = cost

def gradient_descent_for_test(alpha):
    while True:
        # alpha: learning rate
        descent(alpha)
        print cost_function()

def main():
    alpha = 1.0

    # tain
    print 'Loading train set...'
    load_from_file('train_temp.csv')
    print 'Prepare for feature scaling...'
    feature_scaling_init()
    print 'Executing gradient descent...'
    gradient_descent(alpha)

    fout = open('result-theta.txt', 'a')
    fout.write(str(theta) + '\n')
    fout.close()

if __name__ == "__main__":
    main()