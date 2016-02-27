#-*-coding:utf8-*-

version = '06'

def division(divisor, divident):
    if divident == 0:
        return 0.0
    else:
        return divisor / divident

def main():
    theta = list()
    featureMean = list()
    featureRange = list()

    # load featureMean and featureRange
    with open('featureMean.txt', 'r') as fin:
        content = fin.read()[1:-1]
        featureMean = [(float) (num) for num in content.split(',')]

    with open('featureRange.txt', 'r') as fin:
        content = fin.read()[1:-1]
        featureRange = [(float) (num) for num in content.split(',')]

    # load theta
    with open('theta_to_gen_result.txt', 'r') as fin:
        content = ''.join(fin.read().split())[1:-1]
        theta = [(float) (feature) for feature in content.split(',')]


    fout = open('result-' + version + '.csv', 'ab')
    fout.write('Id,reference\n')

    # load features list frome test samples
    with open('test_temp.csv') as fin:
        datas = fin.readlines()[1:]

        dimension = len(theta) - 1
        for data in datas:
            data = data.split(',')
            features = [(float) (data[i]) for i in range(1, dimension + 1)]
            
            estimate = theta[dimension]
            for i in range(dimension):
                scaledFeature = division(features[i] - featureMean[i], featureRange[i])
                estimate += theta[i] * scaledFeature

            fout.write(data[0] + ',' + str(round(estimate, 6)) + '\n')

    fout.close()

    fout = open('log.txt', 'ab')
    fout.write('version: ' + version + '\ntheta: ')
    fout.write(str(theta) + '\n\n')
    fout.close()

if __name__ == "__main__":
    main()