# -*- coding: utf-8 -*-
import random

baseDir = "/mnt/hgfs/VMWareShare/"
resDirName = "20190917-IDNER+ASK-sort-T7-JSON"

flagOfDemo = True

flagOfPos = True
flagOfFilter = False
flagOfStatePLON = False
flagOfCIS = True
flagOfQS = True
flagOfConfDiff = True
flagOfOverlap = True
flagOfClusterAvg = True
flagOfRewardRate = False
flagOfStat = False
flagOfProbsIntoState=False

auxRewardScale = 0
# auxRewardScale = 1

# discount = 0.8
# discount = 0.9
discount = 1.0

# expName = "20190917-IDNER-gamma=1.0-qf=0.01-Round2"
# flagOfAsk = False

# expName = "20190917-IDNER+ASK-gamma=1.0-qf=0.01"
# expName = "20190917-IDNER+ASK-gamma=1.0-qf=0.01-sort-T7-Run2"
# expName = "20190917-IDNER+ASK-gamma=1.0-qf=0.1-sort-T7-Run1"
expName = "20200428"
# expName = "20190917-IDNER+ASK-gamma=1.0-qf=0.01-WT4"
# expName = "20190917-IDNER+ASK-gamma=1.0-qf=0.01-sort-WT4"
port = 50000

##########################################################
# Standard
flagOfPropagation=True
flagOfSelection=True
selectionStrategy = "rl"
##########################################################


##########################################################
queryType=None
# queryType="Identical"
# queryType="Super"
# queryType="Sub"
##########################################################

# humanCost = 1.0
# humanCost = 0.64
# humanCost = 0.32
# humanCost = 0.16
humanCost = 0.08
# humanCost = 0.04
# humanCost = 0.02
# humanCost = 0.01
# humanCost = 0.0

# humanCost = 0.24 # CoNLL only

##########################################################
# RL selection
# flagOfPropagation=False
# flagOfSelection=True
# selectionStrategy = "rl"
##########################################################


##########################################################
# LeastConfidence Selection
# flagOfPropagation=True
# flagOfSelection=False
# selectionStrategy = "confidence"

# preHumanRatio = 0.1
# preHumanRatio = 0.2
# preHumanRatio = 0.3
# preHumanRatio = 0.4
# preHumanRatio = 0.5
# preHumanRatio = 0.6
# preHumanRatio = 0.7
# preHumanRatio = 0.8
# preHumanRatio = 0.9
##########################################################


# dataset = "AKWS1News_E_F4"
dataset = "CoNLLTest_RR_F4A"
# dataset = "Ontonotes_EN_Sub_R_F4B"

# runSeq = "1"
# runSeq = "2"
runSeq = "3"
# runSeq = "4"
# runSeq = "5"
# runSeq = "6"


# 在开发集和测试集上运行50回合
# 实际上只为在开发集上获得最大目标值，不需要在测试集上运行
lrVersion = "1"
# 当目标值在开发集上达到最优值，减少学习率
# lrVersion = "2"

if lrVersion=="1":
    lrStart = 0.000025  # 2.5 * 10^-5
    lrStr = "lrs1[" + str(lrStart) + "]"
elif lrVersion=="2":
    lrStart = 0.000025
    numOfLRDec = 5 # todo: ->2
    numOfEpochsAfterLRDec = 5
    lrStr = "lrs2[" + str(lrStart) + "," + str(numOfLRDec) + "," + str(numOfEpochsAfterLRDec) + "]"
else:
    raise RuntimeError("unknown lrVersion")

qfPenalty = 0.01
queryCost = qfPenalty
finishCost = qfPenalty


penalty = 0
# penalty = 0.01

# expName = "20190328-14-T7-IDNER"
# port = 7000

# expName = "20190411-IDNER"
# port = 17000

# expName = "20190412-IDNER-STATE-PLON"
flagOfStatePLON = True

# expName = "20190420-IDNER"
# port = 20000




# classifierModel='Bagging'
# classifierModel="DT_J48"
# classifierModel='RandomForest'
classifierModel='SVM_LibSVM_RBF'
# classifierModel='LogisticRegression'
# classifierModel='NaiveBayes'
# classifierModel='MultilayerPerceptron'


print "-------------------------------"
print "expName = ", expName
print "-------------------------------"
if flagOfFilter:
    expName+="-filter"

maxNumOfMentionsPerDoc = None

expName += "-Run" + runSeq
expName += "-" + lrStr


if queryType:
    expName += "-Query="+queryType

if not flagOfSelection:
    expName += "-Selection="+selectionStrategy
if not flagOfPropagation:
    expName += "-NoPropagation"

if flagOfDemo:
    port = 50000
else:
    port = random.choice(range(10000, 20000, 1))















posTags = [
"START",    # 0
":",
"CD",
"NNP",
"(",
"IN",       # 5
".",
"DT",
"CC",
",",
"VBD",		# 10
"JobTitle",
"TO",
"VB",
"RP",
"``",		# 15
")",
"JJ",
"VBZ",
"POS",
"NN",		# 20
"WDT",
"NNS",
"VBG",
"VBN",
"VBP",		# 25
"WRB",
"PRP$",
"JJS",
"RB",
"WP$",		# 30
"JJR",
"PRP",
"RBR",
"$",
"WP",		# 35
"MD",
"''",
"FW",
"SYM",
"NNPS",		# 40
"#",
"EX",
"LS",
"UH",
"END",		# 45
"RBS",
"PDT"
]


