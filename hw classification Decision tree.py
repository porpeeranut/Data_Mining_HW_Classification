import numpy as np
import math, sys, getopt, os, fileinput, copy, random
from aenum import Enum

#   Decision tree

def usage():
    fileName = os.path.basename(sys.argv[0])
    print "\nusage: ", fileName,
    print "[option]"
    print " -c      : to test 10%s cross validation" % ('%')
    print " -t  arg : arg is training set file"
    print " -r  arg : arg is attribute range"
    print "\nex."
    print fileName, " -t iris.data -r 12 -c"

class Tree(object):

    class Type(Enum):
        AttrbName = 1
        AttrbValue = 2
        Class = 3

    def __init__(self, data, ttype):
        self.data = data
        self.type = ttype
        self.branch = []

class DecisionTree:

    def __init__(self):
        self.tree = None

    def entropy(self, classOutCnt):
        ent = 0.0
        count = 0.0
        for key, value in classOutCnt.iteritems():
            count += value
        for key, value in classOutCnt.iteritems():
            ent -= (value/count)*math.log(value/count, 2)
        return ent

    def info(self, attribValCnt):
        info = 0.0
        count = 0.0
        subCount = {}
        for key, value in attribValCnt.iteritems():
            for subK, subV in value.iteritems():
                count += subV
                if key in subCount:
                    subCount[key] += subV
                else:
                    subCount[key] = subV
        for key, value in attribValCnt.iteritems():
            info += (subCount[key]/count)*self.entropy(value)
        return info

    def print_tree(self, trainingFile):
        self.print_tree_recur_fn(trainingFile, self.tree)

    def print_tree_recur_fn(self, trainingFile, tree, level=1):
        if tree.type==Tree.Type.AttrbName:
            if trainingFile == "test.data":
                if int(tree.data) == 0:
                    print 'age'
                elif int(tree.data) == 1:
                    print 'income'
                elif int(tree.data) == 2:
                    print 'student'
                elif int(tree.data) == 3:
                    print 'credit_rating'
            else:
                print "attrib", tree.data
        elif tree.type==Tree.Type.AttrbValue:
            print "value", tree.data.split("-")[1]
        elif tree.type==Tree.Type.Class:
            if trainingFile == "iris.data":
                if int(tree.data) == 0:
                    print 'Iris-setosa'
                elif int(tree.data) == 1:
                    print 'Iris-versicolor'
                elif int(tree.data) == 2:
                    print 'Iris-virginica'
            else:
                if int(tree.data) == 0:
                    print 'no'
                elif int(tree.data) == 1:
                    print 'yes'

        for i in range(len(tree.branch)):
            print "\t"*(level-1), "|"
            print "\t"*(level-1), "'------",
            self.print_tree_recur_fn(trainingFile, tree.branch[i], level+1)

    def gen_tree(self, xin):
        self.tree = self.gen_sub_tree(xin)
        return self.tree

    def gen_sub_tree(self, xin):
        classOutCnt = {}
        attribValCnt = {}
        for x in xin:
            clsName = int(x[x.keys()[-1]])
            for key, value in x.iteritems():
                subKey = int(value)
                if key in attribValCnt:
                    if subKey in attribValCnt[key]:
                        if clsName in attribValCnt[key][subKey]:
                            attribValCnt[key][subKey][clsName] += 1
                        else:
                            attribValCnt[key][subKey][clsName] = 1
                    else:
                        attribValCnt[key][subKey] = {clsName: 1}
                else:
                    attribValCnt[key] = {subKey: {clsName: 1}}

            if clsName in classOutCnt:
                classOutCnt[clsName] += 1
            else:
                classOutCnt[clsName] = 1

        if len(classOutCnt) == 1:
            # add class to this node (pure subset)
            return Tree(str(classOutCnt.keys()[0]), Tree.Type.Class)

        ent = self.entropy(classOutCnt)
        mostGainVal = 0.0
        mostGainAttrib = -1
        ks = attribValCnt.keys()
        ks.pop(len(ks)-1)
        for key in ks:
            info = self.info(attribValCnt[key])
            gain = ent - info
            if mostGainVal < gain:
                mostGainVal = gain
                mostGainAttrib = key

        if mostGainAttrib == -1:
            return Tree(str(self.findKeyOfMostClassAmount(classOutCnt)), Tree.Type.Class)

        if len(xin[0]) == 2:
            # have just 2 attrib (feature, class)            
            return Tree(str(self.findKeyOfMostClassAmount(classOutCnt)), Tree.Type.Class)

        root = Tree(mostGainAttrib, Tree.Type.AttrbName)
        
        #   create new input x for each attrib value
        newInput = {}
        for x in xin:
            r = int(x[mostGainAttrib])
            x.pop(mostGainAttrib)
            if r in newInput:
                newInput[r].append(x)
            else:
                newInput[r] = []
                newInput[r].append(x)
        for r, value in attribValCnt[mostGainAttrib].iteritems():
            brnch = Tree(str(mostGainAttrib)+"-"+str(r), Tree.Type.AttrbValue)
            brnch.branch.append(self.gen_tree(newInput[r]))
            root.branch.append(brnch)
        
        return root

    def findKeyOfMostClassAmount(self, classOutCnt):
        classMax = 0
        classKey = 0
        for key, value in classOutCnt.iteritems():
            if classMax < value:
                classMax = value
                classKey = key
        return classKey

    def test(self, listX, trainingFile):
        print "\n------------Testing-------------"
        print "\nFeatures",
        if trainingFile == "iris.data":
            print "\t\tOutput\t\t\tDesired class"
        else:
            print "\t\t\tOutput\t\tDesired class"
        correct = 0
        np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
        for x in listX:
            ks = x.keys()
            ks.pop(len(ks)-1)
            for k in ks:
                print x[k],
            print "\t\t",

            desire = int(x[x.keys()[-1]])
            try:
                out = int(self.predict(x, self.tree))
            except:
                out = -1
            if out == desire:
                correct += 1

            if trainingFile == "iris.data":
                if out == 0:
                    print 'Iris-setosa\t\t',
                elif out == 1:
                    print 'Iris-versicolor\t\t',
                elif out == 2:
                    print 'Iris-virginica\t\t',
                else:
                    print '--------------\t\t',

                if desire == 0:
                    print 'Iris-setosa'
                elif desire == 1:
                    print 'Iris-versicolor'
                elif desire == 2:
                    print 'Iris-virginica'
                else:
                    print "---------------", desire
            else:
                if out == 0:
                    print 'no\t\t',
                elif out == 1:
                    print 'yes\t\t',
                else:
                    print '--------------\t\t',

                if desire == 0:
                    print 'no'
                elif desire == 1:
                    print 'yes'
                else:
                    print "---------------", desire
            
        print "\nAccuracy %.4f%s" % (correct/(len(listX)*1.0)*100.0, '%')
        return correct/(len(listX)*1.0)*100.0

    def predict(self, x, tree):
        if tree.type==Tree.Type.AttrbName:
            for b in tree.branch:
                if x[tree.data] == int(b.data.split("-")[1]):
                    return self.predict(x, b.branch[0])
        elif tree.type==Tree.Type.Class:
            return tree.data

def main(argv):
    isCrossValid = 0
    trainingFile = '-'
    maxAttribRange = 2
    try:
        opts, args = getopt.getopt(argv,"cht:r:")
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
        elif opt in ("-c"):
            isCrossValid = 1
        elif opt in ("-t"):
            trainingFile = arg
        elif opt in ("-r"):
            maxAttribRange = int(arg)

    listX = []
    shuffleX = []
    if trainingFile == "iris.data":
        with open(trainingFile) as f:
            minF = [10, 10, 10, 10]
            maxF = [0, 0, 0, 0]
            for line in f:
                tmp = line.split(',')
                tmp.pop()
                tmp = map(float, tmp)
                for j in range(len(tmp)):
                    if minF[j] > tmp[j]:
                        minF[j] = tmp[j]
                    if maxF[j] < tmp[j]:
                        maxF[j] = tmp[j]

        with open(trainingFile) as f:
            for line in f:
                tmp = line.split(',')
                if tmp[4].strip() == 'Iris-setosa':
                    tmp[4] = 0
                elif tmp[4].strip() == 'Iris-versicolor':
                    tmp[4] = 1
                elif tmp[4].strip() == 'Iris-virginica':
                    tmp[4] = 2

                tmp = map(float, tmp)
                d = {}
                for j in range(len(tmp)-1):
                    for r in range(maxAttribRange):
                        if tmp[j] <= minF[j]+(r+1)*((maxF[j]-minF[j])/maxAttribRange):
                            d[j] = r
                            break
                d[len(tmp)-1] = tmp[-1]
                listX.append(d)

            rdIndex = random.sample(range(len(listX)), len(listX))
            for i in rdIndex:
                shuffleX.append(listX[i])
            inputX = np.array(shuffleX)
    if trainingFile == "test.data":
        with open(trainingFile) as f:
            minF = [10, 10, 10, 10]
            maxF = [0, 0, 0, 0]
            for line in f:
                tmp = line.split(',')
                tmp.pop()
                tmp = map(float, tmp)
                for j in range(len(tmp)):
                    if minF[j] > tmp[j]:
                        minF[j] = tmp[j]
                    if maxF[j] < tmp[j]:
                        maxF[j] = tmp[j]

        with open(trainingFile) as f:
            for line in f:
                tmp = line.split(',')
                if tmp[4].strip() == 'no':
                    tmp[4] = 0
                elif tmp[4].strip() == 'yes':
                    tmp[4] = 1

                tmp = map(float, tmp)
                d = {}
                for j in range(len(tmp)-1):
                    for r in range(maxAttribRange):
                        d[j] = tmp[j]
                d[len(tmp)-1] = tmp[-1]
                listX.append(d)

            rdIndex = random.sample(range(len(listX)), len(listX))
            for i in rdIndex:
                shuffleX.append(listX[i])
            inputX = np.array(shuffleX)

    dt = DecisionTree()
    if isCrossValid == 1:
        # 10% cross validation
        accAV = 0.0
        for p in range(0, 10):
            print "\n\n------------- Cross validation block", p+1, "-------------"
            block = int(round(len(listX)/10.0, 0))
            end = (p*block+block)-1
            if p == 9:
                end = len(listX)-1
            tmpTestListX = []
            trainListX = copy.deepcopy(shuffleX)
            for i in range(end, p*block-1, -1):
                tmpTestListX.append(trainListX[i])
                trainListX.pop(i)

            testDataX = np.array(tmpTestListX)
            trainDataX = np.array(trainListX)

            dt.gen_tree(copy.deepcopy(trainDataX))
            dt.print_tree(trainingFile)
            accAV += dt.test(testDataX, trainingFile)
            print "----------------------------------------------------"
        print "Accuracy Average %.5f%s" % (accAV/10, '%')
    else:
        print 
        dt.gen_tree(copy.deepcopy(inputX))
        dt.print_tree(trainingFile)
        dt.test(inputX, trainingFile)

if __name__ == "__main__":
    main(sys.argv[1:])