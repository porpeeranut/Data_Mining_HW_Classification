import numpy as np
import math, sys, getopt, os, fileinput, copy, random

#   Neural Networks Backpropagation

def usage():
    fileName = os.path.basename(sys.argv[0])
    print "\nusage: ", fileName,
    print "[option]"
    print " -N  arg : arg is neural name"
    print " -a  arg : arg is activation function \"tanh\" or \"sigmoid\""
    print " -n  arg : arg is learning rate"
    print " -m  arg : arg is momentum"
    print " -e  arg : arg is number of epoch to exit"
    print " -c      : to test 10%s cross validation" % ('%')
    print " -t  arg : arg is training set file"
    print "\nex."
    print fileName, "-N 4-4-3 -a sigmoid -n 0.8 -m 0.01 -e 10000 -t iris.data"  # 99.3333%

class NeuralNetwork:

    def __init__(self, layers, activFunct, learning_rate, momentum, epoch, error):
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.epoch = epoch
        self.error = float(error)
        if activFunct == 'sigmoid':
            self.activation = sigmoid
            self.activation_derive = sigmoid_derivertive
        else:
            self.activation = tanh
            self.activation_derive = tanh_derivertive

        # Init weight
        self.init_weights = []
        for i in range(1, len(layers) - 1):
            fanin = layers[i-1]
            weightInit = 1/math.sqrt(fanin)
            cellPrev = layers[i-1]
            cellCurr = layers[i]
            r = np.random.uniform(-1*weightInit, weightInit, [cellPrev + 1, cellCurr + 1])
            self.init_weights.append(r)
        fanin = layers[i-1]
        weightInit = 1/math.sqrt(fanin)
        cellOutput = layers[i+1]
        cellPreOutput = layers[i]
        r = np.random.uniform(-1*weightInit, weightInit, [cellPreOutput + 1, cellOutput])
        self.init_weights.append(r)
        #print "\nWeights:", self.init_weights

    def train(self, x, y):
        #print "\n------------Training------------"
        # add bias 1 to input layer
        bias = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((bias.T, x), axis=1)

        self.weights = copy.deepcopy(self.init_weights)
        self.old_weights = copy.deepcopy(self.weights)

        for e in range(int(self.epoch) + 1):
            #   print progress
            # if e % int(int(self.epoch)/13) == 0:
            #     #print " Epoch", e, "--", e*10 / int(int(self.epoch)/10), '%'
            #     sys.stdout.write('==')
            # if e == int(self.epoch):
            #     print "%d%s" % (e*10 / int(int(self.epoch)/10), '%')
            
            for i in range(int(x.shape[0])):
                y_all = [x[i]]
                v_all = [[]]
                # feedforward networks
                for l in range(len(self.weights)):
                    v_layer = np.dot(y_all[l], self.weights[l])
                    y_layer = self.activation(v_layer)
                    v_all.append(v_layer)
                    y_all.append(y_layer)

                # gradients at output layer
                error = y[i] - y_all[-1]
                # if e % 100 == 0 and i == 0:
                #     print error, e
                gradients = [error * self.activation_derive(v_all[-1])]

                # gradients at hidden layer
                for l in range(len(y_all)-2, 0, -1):
                    gradients.append(self.activation_derive(v_all[l])*gradients[-1].dot(self.weights[l].T))
                gradients.reverse()

                # set new weight for back propagation
                self.tmp_old_weights = copy.deepcopy(self.weights)
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(y_all[i])
                    gradient = np.atleast_2d(gradients[i])
                    delta_weight = self.weights[i] - self.old_weights[i]
                    self.weights[i] += self.momentum * delta_weight + self.learning_rate * layer.T.dot(gradient)

                self.old_weights = copy.deepcopy(self.tmp_old_weights)
        for i in range(len(self.weights)):
            print self.weights[i]

    def test(self, listX, listY, trainingFile):
        print "\n------------Testing-------------"
        print "\nFeatures",
        if trainingFile == "iris.data":
            print "\t\tOutput\t\t\t\t\t\tDesired class"
        else:
            print "\t\tOutput\t\t\tDesired class"
        EsumSqr = 0
        correct = 0
        i = 0
        np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
        for x in listX:
            if trainingFile == "iris.data":
                print " ".join('%0.1f' % f for f in x), "\t",
            elif trainingFile == "cross.pat":
                print " ".join('%0.4f' % f for f in x), "\t",
            else:
                print " ".join('%d' % f for f in x), "\t\t\t",

            # add bias 1 to input layer
            x = np.concatenate((np.ones(1), np.array(x)))
            for l in range(0, len(self.weights)):
                v = np.dot(x, self.weights[l])
                x = self.activation(v)
            error = listY[i] - x[-1]
            Esum = 0;
            if isinstance(error, np.float64):
                Esum += error**2
            else:
                for e in error:
                    Esum += e**2
            EsumSqr += Esum/2
            #print "error",error
            desireY = listY[i]
            tmp = x.tolist()
            idx = tmp.index(max(tmp))
            if idx == 0:
                out = np.array([1, 0, 0])
            elif idx == 1:
                out = np.array([0, 1, 0])
            elif idx == 2:
                out = np.array([0, 0, 1])

            if trainingFile == "iris.data":
                if (desireY==out).all():
                    correct = correct+1
            elif trainingFile == "cross.pat":
                if x[0] > x[1]:
                    out = np.array([1, 0])
                else:
                    out = np.array([0, 1])
                if (out == listY[i]).all():
                    correct = correct+1
                out = x
            else:
                if (x < 0.5 and listY[i] == 0) or (x >= 0.5 and listY[i] == 1):
                    correct = correct+1
                out = x

            print x, "\t",
            if trainingFile == "iris.data":
                #print "%d" % (desireY)
                print desireY
            else:
                print desireY
            i = i+1
        Eav = EsumSqr/len(listX)
        print "\nError Avg %.10f" % (Eav)
        print "Accuracy %.5f%s" % (correct/(len(listY)*1.0)*100.0, '%')
        return Eav, correct/(len(listY)*1.0)*100.0

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivertive(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivertive(x):
    return (1.0/np.cosh(x))**2

def main(argv):
    NNnameList = []
    learning_rate = 0.2
    momentum = 0.1
    epoch = 1000
    error = 0.001
    isCrossValid = 0
    activFunct = 'tanh'
    trainingFile = '-'
    try:
        opts, args = getopt.getopt(argv,"chN:a:n:m:e:E:t:")
        if len(sys.argv) == 1:
            usage()
            sys.exit(2)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-N"):
            NNname = arg
            NNnameList = arg.split('-')
            NNnameList = map(int, NNnameList)
        elif opt in ("-a"):
            activFunct = arg
        elif opt in ("-n"):
            learning_rate = arg
        elif opt in ("-m"):
            momentum = arg
        elif opt in ("-e"):
            epoch = arg
        elif opt in ("-E"):
            error = arg
        elif opt in ("-c"):
            isCrossValid = 1
        elif opt in ("-t"):
            trainingFile = arg

    print "\n------------Variable------------"
    print 'Neural name\t', NNname
    print 'Activation func\t', activFunct
    print 'Learning rate\t', learning_rate
    print 'Momentum\t', momentum
    print 'Epoch\t\t', epoch
    #print 'Min error\t', error
    print 'TrainingFile\t', trainingFile

    nn = NeuralNetwork(NNnameList, activFunct, learning_rate, momentum, epoch, error)

    listX = []
    listY = []
    shuffleX = []
    shuffleY = []
    if trainingFile == "iris.data":
        with open(trainingFile) as f:
            for line in f:
                tmp = line.split(',')
                if tmp[4].strip() == 'Iris-setosa':
                    listY.append([1, 0, 0])
                elif tmp[4].strip() == 'Iris-versicolor':
                    listY.append([0, 1, 0])
                elif tmp[4].strip() == 'Iris-virginica':
                    listY.append([0, 0, 1])                    
                tmp.pop()
                tmp = map(float, tmp)
                listX.append(tmp)

            rdIndex = random.sample(range(len(listX)), len(listX))
            for i in rdIndex:
                shuffleX.append(listX[i])
                shuffleY.append(listY[i])
            inputX = np.array(shuffleX)
            outputY = np.array(shuffleY)
    else:
        inputX = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

        outputY = np.array([0, 1, 1, 0])

    # print "\nInput X"
    # print inputX
    # print "\nDesire output"
    # print outputY

    if isCrossValid == 1:
        # 10% cross validation
        errorAV = 0.0
        accAV = 0.0
        for p in range(0, 10):
            print "\n\n------------- Cross validation block", p+1, "-------------"
            block = int(round(len(listX)/10.0, 0))
            end = (p*block+block)-1
            if p == 9:
                end = len(listX)-1
            tmpTestListX = []
            tmpTestListY = []
            trainListX = copy.deepcopy(shuffleX)
            trainListY = copy.deepcopy(shuffleY)
            for i in range(end, p*block-1, -1):
                tmpTestListX.append(trainListX[i])
                tmpTestListY.append(trainListY[i])
                trainListX.pop(i)
                trainListY.pop(i)

            testDataX = np.array(tmpTestListX)
            testDataY = np.array(tmpTestListY)
            trainDataX = np.array(trainListX)
            trainDataY = np.array(trainListY)
            nn.train(trainDataX, trainDataY)
            err, acc = nn.test(testDataX, testDataY, trainingFile)
            errorAV += err 
            accAV += acc
            print "----------------------------------------------------"
        print "Error Average ", errorAV/10
        print "Accuracy Average %.5f%s" % (accAV/10, '%')
    else:
        nn.train(inputX, outputY)
        nn.test(inputX, outputY, trainingFile)

if __name__ == "__main__":
    main(sys.argv[1:])