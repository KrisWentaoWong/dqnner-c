# -*- coding: utf-8 -*-
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import zmq, time
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
import argparse
from random import shuffle
from operator import itemgetter
import matplotlib.pyplot as plt
import constants
import util
import dataTypes

import random

import fileUtil

#DEBUG = False
ANALYSIS = False
COUNT_ZERO = False

#Global variables
resDir = constants.baseDir + constants.resDirName + "/" + constants.dataset + "/" + constants.classifierModel + "/"
outDir = constants.baseDir + constants.expName + "/" + constants.dataset + "/" + constants.classifierModel + "/"

fileUtil.mkdir(outDir)

if constants.lrVersion=="2":
    maxObjectiveInLRStudysFile = outDir + "maxObjectiveInLRStudys.txt"
    maxObjectiveInLRStudys  = fileUtil.readToObject(maxObjectiveInLRStudysFile)
    maxObjectiveInLRStudy = maxObjectiveInLRStudys[str(constants.humanCost)]
    print "maxObjectiveInLRStudy=", maxObjectiveInLRStudy

if constants.queryType:
    resDir += "query=" + constants.queryType + "/"

if constants.flagOfSelection:
    outDir += "HC=" + str(constants.humanCost) + "/"
elif constants.selectionStrategy == "random":
    outDir += "RR=" + str(constants.randomRatio) + "/"
elif constants.selectionStrategy == "confidence":
    print "preHumanRatio=", constants.preHumanRatio
    resDir += "preHumanRatio=" + str(constants.preHumanRatio) + "/"
    outDir += "preHumanRatio=" + str(constants.preHumanRatio) + "/"



fileUtil.mkdir(outDir)

detailDir_test = outDir + "detail/"
fileUtil.mkdir(detailDir_test)

detailDir_dev = outDir + "detail.dev/"
fileUtil.mkdir(detailDir_dev)

print "dataset=", constants.dataset
print "classifierModel=", constants.classifierModel
print "outDir=", outDir
print "detailDir_test=", detailDir_test
print "detailDir_dev=", detailDir_dev
# print "flagOfQS=", constants_idner.flagOfQS
# print "flagOfCIS=", constants_idner.flagOfCIS
# print "flagOfStatePLON=", constants_idner.flagOfStatePLON
print "penalty=", constants.penalty
print "*** humanCost=", constants.humanCost, " ***"
print "*** queryCost=", constants.queryCost
print "*** discount=", constants.discount
# print "auxRewardScale=", constants_idner.auxRewardScale

NUM_NETYPES = 4 # PER, LOC, ORG, O
STATE_SIZE = 19 + 3*3*2
if constants.flagOfStatePLON:
    STATE_SIZE += NUM_NETYPES * 2
if constants.flagOfPos:
    STATE_SIZE += len(constants.posTags) * 4 * 2

if constants.flagOfStat:
    # rateFactor
    STATE_SIZE += 1
    # stat 7
    STATE_SIZE += 1+6

if constants.flagOfProbsIntoState:
    STATE_SIZE += 9+9

print "state_size=", STATE_SIZE

ACCEPT = 1
REJECT = 2
# ASK = 3
# FINISH = 9
mapOfIntToActionName = {
    ACCEPT: "ACCEPT",
    REJECT: "REJECT",
}

QUERY_IDENTICAL = 1
QUERY_SUPER = 2
QUERY_SUB = 3
ASK = 4
FINISH = 5
mapOfIntToQueryName = {
    QUERY_IDENTICAL: "Identical",
    QUERY_SUPER: "Super",
    QUERY_SUB: "Sub",
    ASK: "ASK",
    # Keep finish as the last query, because we force the query to be the last query (finish) when count of steps exceeds.
    FINISH: "FINISH"
}

QUERIES_FOR_SIMILAR_MENTIONS = [QUERY_IDENTICAL, QUERY_SUPER, QUERY_SUB]


def dd():
    return {}               # lutm: 小括号 元组；中括号 列表；大括号 键值对map

def ddd():
    return collections.defaultdict(dd)

numOfEvalingCorrect = 0
numOfEvalingGold = 0
numOfEvaling = 0
numOfHumanTT = 0
numOfHumanFT = 0
QUERY = collections.defaultdict(lambda:0.)
ACTION = collections.defaultdict(lambda:0.)
CHANGES = 0
evalMode = False
STAT_POSITIVE, STAT_NEGATIVE = 0, 0 #stat. sign.

CONTEXT = None

docLevelErrorRates = []
docLevelHumanRates = []

def splitBars(w):
    return [q.strip() for q in w.split('|')]

#Environment for each episode
class Environment:
    # srcdoc: boundary
    # extdocs: a[qIdx][dldIdx]
    # goldEntities: [label]
    # docIdx
    # evalMode
    def __init__(self, doc, docIdx, evalMode):
        doc = copy.deepcopy(doc)
        self.doc = doc
        self.mentionsInDoc = doc["mentions"]
        self.mapOfKeyToMention = util.constructMapOfKeyToMention(self.mentionsInDoc)
        # assert self.mapOfNameToCluster["hasQueriedHuman"]==False

        self.numOfMentionsInDoc = len(self.mentionsInDoc)
        # self.numOfMentionsInDoc_norm = self.numOfMentionsInDoc*1.0/constants_idner.maxNumOfMentionsPerDoc
        self.rateFactor = 1.0/self.numOfMentionsInDoc

        self.finishedMentions = []
        self.unfinishedMentions = copy.copy(self.mentionsInDoc)
        self.humanMentions = []

        self.docIdx = docIdx

        # temp
        self.idxInState = None
        self.stateStr = None

        # self.probChangeOfEpisode = 0.0
        self.currentMentionIdx = self.findNextUnannotatedMentionIdx(-1)
        self.stepSeq = 1
        self.stepSeqForCurrentMention = 1

        self.lastAction = None
        self.terminal = False

        self.availbleOfIdentical = False
        self.availbleOfSuper = False
        self.availbleOfSub = False

        # use last query to construct current state
        # then the agent select current action and query
        self.lastQuery = self.selectInitialQuery()
        self.constructState(self.lastQuery)

        return


    def findNextUnannotatedMentionIdx(self, currentMentionIdx):
        nextIdx = currentMentionIdx + 1

        while nextIdx < self.numOfMentionsInDoc:
            nextMention = self.mentionsInDoc[nextIdx]
            if not nextMention["hasQueriedHuman"]:
                return nextIdx
            nextIdx += 1

        return nextIdx


    def selectInitialQuery(self):
        currentMention = self.mentionsInDoc[self.currentMentionIdx]

        if evalMode and DEBUG:
            detailOutFile.write("\n\nSelectInitialQuery")
        queriesHavingResults = []
        for query in QUERIES_FOR_SIMILAR_MENTIONS:
            queryName = mapOfIntToQueryName[query]
            mapOfMentionKeysByQueryNames = currentMention["mapOfMentionKeysByQueryNames"]
            mentionKeysByQuery = mapOfMentionKeysByQueryNames[queryName]
            if evalMode and DEBUG:
                detailOutFile.write("\t" + str(len(mentionKeysByQuery)))
            if len(mentionKeysByQuery)>0:
                queriesHavingResults.append(query)

        if len(queriesHavingResults)==0:
            initialQuery = random.choice(QUERIES_FOR_SIMILAR_MENTIONS)
            if evalMode and DEBUG:
                detailOutFile.write("\nInitial query: no available, use " + str(initialQuery))
        else:
            initialQuery = random.choice(queriesHavingResults)
            if evalMode and DEBUG:
                detailOutFile.write("\nInitial query: use " + str(initialQuery))

        return initialQuery


    def constructState(self, lastQuery):
        currentMention = self.mentionsInDoc[self.currentMentionIdx]
        if self.stepSeqForCurrentMention == 1:
            self.initMentionInFirstStep(currentMention)

        lastQueryName = mapOfIntToQueryName[lastQuery] if lastQuery != None else "EMPTY_QUERY"
        self.mentionByLastQuery = self.getMentionByLastQuery(lastQuery, currentMention)
        mentionByLastQuery = self.mentionByLastQuery
        # [optional] update attributes related to currentMention
        if DEV:
            temp = "mentionByLastQuery="
            if mentionByLastQuery == None:
                temp += "None"
            else:
                temp += mentionByLastQuery["boundaryKey"] + mentionByLastQuery["name"]
            print(temp)

        stateDesc = ""
        if evalMode and DEBUG:
            stateDesc += "\n"
            stateDesc += "\n---------------- STEP " + str(self.stepSeq) + ", MENTION " + str(self.currentMentionIdx+1) + "." + str(self.stepSeqForCurrentMention) + "----------------"
            stateDesc += "\nMention: "+mentionToStr(currentMention)
            stateDesc += "\nByQuery: "+mentionToStr(mentionByLastQuery)
            detailOutFile.write(stateDesc)

        self.state = [0 for i in range(STATE_SIZE)]
        self.stateStr = ""
        self.idxInState = -1

        if constants.flagOfStat:
            ################################# state: stat
            if evalMode and DEBUG:
                self.stateStr += "\n\tStat:"
            # total normalized
            self.addStateAttr(self.numOfMentionsInDoc_norm)
            self.addStateAttr(self.rateFactor)

            # progress normalized (/constants_idner.maxNumOfMentionsPerDoc)
            if evalMode and DEBUG:
                self.stateStr += "\n\tprogress normalized:"
            self.addStateAttr(len(self.finishedMentions) * 1.0 / constants.maxNumOfMentionsPerDoc)
            self.addStateAttr((self.numOfMentionsInDoc-len(self.finishedMentions)) * 1.0 / constants.maxNumOfMentionsPerDoc)
            if evalMode and DEBUG:
                self.stateStr += "\t[H]:"
            self.addStateAttr(len(self.humanMentions) * 1.0 / constants.maxNumOfMentionsPerDoc)

            if evalMode and DEBUG:
                self.stateStr += "\n\tprogress percentage:"
            # progress percentage (/self.numOfMentionsInDoc)
            self.addStateAttr(len(self.finishedMentions)*1.0/self.numOfMentionsInDoc)
            self.addStateAttr(1-(len(self.finishedMentions)*1.0/self.numOfMentionsInDoc))
            if evalMode and DEBUG:
                self.stateStr += "\t[H]:"
            self.addStateAttr(len(self.humanMentions)*1.0/self.numOfMentionsInDoc)


        if constants.flagOfProbsIntoState:
            self.addProbsIntoState()


        ################################# state: current mention
        if evalMode and DEBUG:
            self.stateStr += "\n\tcurrentMention:"
        self.addStateAttr(currentMention["hasQueriedHuman"], dataTypes.BOOLEAN, "\tH[%d]")
        self.addStateAttr(currentMention["prob"])
        self.addStateAttr(currentMention["probOfContainsPLO"   ])
        self.addStateAttr(currentMention["probOfContainsPer"   ])
        self.addStateAttr(currentMention["probOfContainsLoc"   ])
        self.addStateAttr(currentMention["probOfContainsOrg"   ])
        self.addStateAttr(currentMention["probOfContainedByPLO"])
        self.addStateAttr(currentMention["probOfContainedByPer"])
        self.addStateAttr(currentMention["probOfContainedByLoc"])
        self.addStateAttr(currentMention["probOfContainedByOrg"])
        self.addStateAttr(currentMention["probOfOverlapWithPLO"])
        self.addStateAttr(currentMention["probOfOverlapWithPer"])
        self.addStateAttr(currentMention["probOfOverlapWithLoc"])
        self.addStateAttr(currentMention["probOfOverlapWithOrg"])
        if constants.flagOfStatePLON:
            self.addStateAttr(currentMention["neType"]=="PER", dataTypes.BOOLEAN)
            self.addStateAttr(currentMention["neType"]=="LOC", dataTypes.BOOLEAN)
            self.addStateAttr(currentMention["neType"]=="ORG", dataTypes.BOOLEAN)
            self.addStateAttr(currentMention["neType"]=="O"  , dataTypes.BOOLEAN)
        if constants.flagOfPos:
            self.addPosTagToState(currentMention, "prevPosTag1")
            self.addPosTagToState(currentMention, "startPosTag")
            self.addPosTagToState(currentMention, "endPosTag")
            self.addPosTagToState(currentMention, "succPosTag1")

        ################################# state: mention by query
        if evalMode and DEBUG:
            self.stateStr += "\n\tmByLastQuery:"
        self.addStateAttr(mentionByLastQuery["hasQueriedHuman"     ] if mentionByLastQuery else False, dataTypes.BOOLEAN, "\tH[%d]")
        # self.addStateAttr(mentionByLastQuery["prob"                ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probAsEvi"           ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainsPLO"   ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainsPer"   ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainsLoc"   ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainsOrg"   ] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainedByPLO"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainedByPer"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainedByLoc"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfContainedByOrg"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfOverlapWithPLO"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfOverlapWithPer"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfOverlapWithLoc"] if mentionByLastQuery else 0.0)
        self.addStateAttr(mentionByLastQuery["probOfOverlapWithOrg"] if mentionByLastQuery else 0.0)
        if constants.flagOfStatePLON:
            self.addStateAttr(mentionByLastQuery["neType"]=="PER" if mentionByLastQuery else False, dataTypes.BOOLEAN)
            self.addStateAttr(mentionByLastQuery["neType"]=="LOC" if mentionByLastQuery else False, dataTypes.BOOLEAN)
            self.addStateAttr(mentionByLastQuery["neType"]=="ORG" if mentionByLastQuery else False, dataTypes.BOOLEAN)
            self.addStateAttr(mentionByLastQuery["neType"]=="O"   if mentionByLastQuery else False, dataTypes.BOOLEAN)
        if constants.flagOfPos:
            self.addPosTagToState(mentionByLastQuery, "prevPosTag1")
            self.addPosTagToState(mentionByLastQuery, "startPosTag")
            self.addPosTagToState(mentionByLastQuery, "endPosTag")
            self.addPosTagToState(mentionByLastQuery, "succPosTag1")

        # lastQuery
        if evalMode and DEBUG:
            self.stateStr += "\n\tlastQuery=" + lastQueryName
        self.addStateAttr(lastQuery == QUERY_IDENTICAL, dataTypes.BOOLEAN)
        self.addStateAttr(lastQuery == QUERY_SUPER, dataTypes.BOOLEAN)
        self.addStateAttr(lastQuery == QUERY_SUB, dataTypes.BOOLEAN)

        ################################# state: relation
        if evalMode and DEBUG:
            self.stateStr += "\n"
        # match
        match = currentMention["neType"] == mentionByLastQuery["neType"] if mentionByLastQuery else False
        self.addStateAttr(match, dataTypes.BOOLEAN, "\tmatch=%d")
        # overlap
        if mentionByLastQuery == None:
            overlap = False
        else:
            overlap = util.overlap(currentMention, mentionByLastQuery)
        self.addStateAttr(overlap, dataTypes.BOOLEAN, "\toverlap=%d")
        # diff
        # diff = mentionByLastQuery["prob"] - currentMention["prob"] if mentionByLastQuery else -currentMention["prob"]
        diff = mentionByLastQuery["probAsEvi"] - currentMention["prob"] if mentionByLastQuery else -currentMention["prob"]
        normalizedDiff = (diff / 2) + 0.5
        self.addStateAttr(normalizedDiff, dataTypes.FLOAT, "\tdiff=%0.3f")

        ################################# state: avaliblity
        if evalMode and DEBUG:
            self.stateStr += "\n\tavaliblity: "
        availble = self.availble(currentMention, QUERY_IDENTICAL)
        self.addStateAttr(availble[0], dataTypes.BOOLEAN, "\tI[%d="+availble[1]+"]")
        self.availbleOfIdentical = availble[0]

        availble = self.availble(currentMention, QUERY_SUPER)
        self.addStateAttr(availble[0], dataTypes.BOOLEAN, "\tP[%d="+availble[1]+"]")
        self.availbleOfSuper = availble[0]

        availble = self.availble(currentMention, QUERY_SUB)
        self.addStateAttr(availble[0], dataTypes.BOOLEAN, "\tB[%d="+availble[1]+"]")
        self.availbleOfSub = availble[0]

        ################################# state: cluster
        if evalMode and DEBUG:
            detailOutFile.write("\nSTATE\t=" + self.stateStr)

        return

    def addProbs(self):
        if evalMode and DEBUG:
            self.stateStr += "\n\tprobs   reliable:"
        # probs reliable
        probs = self.calcProbs(self.mentionsInDoc, True)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])
        if evalMode and DEBUG:
            self.stateStr += "\t[F]:"
        probs = self.calcProbs(self.finishedMentions, True)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])
        if evalMode and DEBUG:
            self.stateStr += "\t[U]:"
        probs = self.calcProbs(self.unfinishedMentions, True)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])
        if evalMode and DEBUG:
            self.stateStr += "\n\tprobs unreliable:"
        probs = self.calcProbs(self.mentionsInDoc, False)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])
        if evalMode and DEBUG:
            self.stateStr += "\t[F]:"
        probs = self.calcProbs(self.finishedMentions, False)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])
        if evalMode and DEBUG:
            self.stateStr += "\t[U]:"
        probs = self.calcProbs(self.unfinishedMentions, False)
        self.addStateAttr(probs[0])
        self.addStateAttr(probs[1])
        self.addStateAttr(probs[2])

    def calcProbs(self, mentions, reliable):
        if len(mentions) == 0:
            return [0,0,0]

        min=1
        max=0
        total=0
        for mention in mentions:
            if reliable:
                prob = mention["probAsEvi"]
            else:
                prob = mention["prob"]
            if prob<min:
                min = prob
            if prob>max:
                max = prob
            total += prob
        avg = total*1.0/len(mentions)

        return (min, avg, max)


    def addPosTagToState(self, mention, tokenPosition):
        for posTag in constants.posTags:
            self.idxInState += 1
            self.state[self.idxInState] = 1 if mention and mention[tokenPosition] == posTag else 0
            if evalMode and DEBUG:
                self.stateStr += " " + str(self.state[self.idxInState])

    def addStateAttr(self, value, dataType=dataTypes.FLOAT, formatStr=None):
        self.idxInState += 1
        if dataType == dataTypes.BOOLEAN:
            self.state[self.idxInState] = 1 if value else 0
            if evalMode and DEBUG:
                if formatStr:
                    self.stateStr += formatStr % (self.state[self.idxInState])
                else:
                    self.stateStr += "\t" + str(self.state[self.idxInState])
        elif dataType == dataTypes.FLOAT or dataType == dataTypes.INT:
            self.state[self.idxInState] = value
            if evalMode and DEBUG:
                if formatStr:
                    self.stateStr += formatStr % (self.state[self.idxInState])
                else:
                    self.stateStr += "\t" + format(self.state[self.idxInState] * 1.0, "0.3f")
        else:
            raise RuntimeError()
        return


    # for currentMention, init probOfContainsPLO, probOfContainedByPLO, probOfOverlapWithPLO
    def initMentionInFirstStep(self, currentMention):
        self.updateAttrsRelatedToOtherMentions(currentMention)

        for query in QUERIES_FOR_SIMILAR_MENTIONS:
            queryName = mapOfIntToQueryName[query]
            mapOfMentionKeysByQueryNames = currentMention["mapOfMentionKeysByQueryNames"]
            mentionKeysByQuery = mapOfMentionKeysByQueryNames[queryName]    # sorted in java preprocess
            # some elements may be consulted to human, put these elements to head of the list
            orderedKeys = self.moveConfirmedMentionsToHead(mentionKeysByQuery)
            mapOfMentionKeysByQueryNames[queryName] = orderedKeys
        return

    def moveConfirmedMentionsToHead(self, srcMentionKeys):
        if len(srcMentionKeys)<=1:
            return srcMentionKeys
        confirmedKeys = []
        unconfirmedKeys = []
        for key in srcMentionKeys:
            mention = self.mapOfKeyToMention[key]
            if mention["hasQueriedHuman"]:
                confirmedKeys.append(key)
            else:
                unconfirmedKeys.append(key)
        desKeys = []
        desKeys.extend(confirmedKeys)
        desKeys.extend(unconfirmedKeys)
        return desKeys


    # other mentions related to this mention may have changed, so this mention should be updated
    def updateAttrsRelatedToOtherMentions(self, mention):
        mention.update(probOfContainsPLO=self.calcProbOfPLOAtPosition(mention, "CONTAINS"))
        mention.update(probOfContainedByPLO=self.calcProbOfPLOAtPosition(mention, "CONTAINED_BY"))
        mention.update(probOfOverlapWithPLO=self.calcProbOfPLOAtPosition(mention, "OVERLAP_WITH"))

        mention.update(probOfContainsPer=self.calcProbOfPerLocOrgAtPosition(mention, "probOfPer", "CONTAINS"))
        mention.update(probOfContainsLoc=self.calcProbOfPerLocOrgAtPosition(mention, "probOfLoc", "CONTAINS"))
        mention.update(probOfContainsOrg=self.calcProbOfPerLocOrgAtPosition(mention, "probOfOrg", "CONTAINS"))

        mention.update(probOfContainedByPer=self.calcProbOfPerLocOrgAtPosition(mention, "probOfPer", "CONTAINED_BY"))
        mention.update(probOfContainedByLoc=self.calcProbOfPerLocOrgAtPosition(mention, "probOfLoc", "CONTAINED_BY"))
        mention.update(probOfContainedByOrg=self.calcProbOfPerLocOrgAtPosition(mention, "probOfOrg", "CONTAINED_BY"))

        mention.update(probOfOverlapWithPer=self.calcProbOfPerLocOrgAtPosition(mention, "probOfPer", "OVERLAP_WITH"))
        mention.update(probOfOverlapWithLoc=self.calcProbOfPerLocOrgAtPosition(mention, "probOfLoc", "OVERLAP_WITH"))
        mention.update(probOfOverlapWithOrg=self.calcProbOfPerLocOrgAtPosition(mention, "probOfOrg", "OVERLAP_WITH"))

    def calcProbOfPLOAtPosition(self, currentMention, position):
        mapOfMentionKeysByPositions = currentMention["mapOfMentionKeysByPositions"]
        mentionKeysByPosition = mapOfMentionKeysByPositions[position]
        probOfExistsPLOAtPosition = 0
        for mentionKeyByPosition in mentionKeysByPosition:
            mentionByPosition = self.mapOfKeyToMention[mentionKeyByPosition]
            probOfPLO = util.getProbOfPLO(mentionByPosition)
            if probOfPLO > probOfExistsPLOAtPosition:
                probOfExistsPLOAtPosition = probOfPLO
        return probOfExistsPLOAtPosition

    def calcProbOfPerLocOrgAtPosition(self, currentMention, attrName, position):
        mapOfMentionKeysByPositions = currentMention["mapOfMentionKeysByPositions"]
        mentionKeysByPosition = mapOfMentionKeysByPositions[position]
        probOfExistsPLOAtPosition = 0
        for mentionKeyByPosition in mentionKeysByPosition:
            mentionByPosition = self.mapOfKeyToMention[mentionKeyByPosition]
            probOfPLO = mentionByPosition[attrName]
            if probOfPLO > probOfExistsPLOAtPosition:
                probOfExistsPLOAtPosition = probOfPLO
        return probOfExistsPLOAtPosition

    #################################################### queryMention
    def getMentionByLastQuery(self, lastQuery, currentMention):
        if constants.selectionStrategy == "confidence" and lastQuery == ASK:
            return None

        lastQueryName = mapOfIntToQueryName[lastQuery]
        if lastQuery == QUERY_IDENTICAL or lastQuery == QUERY_SUPER or lastQuery == QUERY_SUB:
            mentionByQuery = self.querySimilarMention(currentMention, lastQuery, lastQueryName)
            if mentionByQuery:
                self.updateAttrsRelatedToOtherMentions(mentionByQuery)
        # elif lastQuery == QUERY_HUMAN:
        #     mentionByQuery = self.queryHumanMention(currentMention)
        # elif lastQuery == QUERY_NONE:
        #     mentionByQuery = None
        else:
            raise RuntimeError()

        return mentionByQuery


    # def queryHumanMention(self, currentMention):
    #     # set all attributes used in state
    #     mentionByQuery = {}
    #
    #     mentionByQuery.update(isHumanMention=True)
    #     mentionByQuery.update(hasQueriedHuman=True)
    #
    #     mentionByQuery.update(docId      =currentMention["docId"])
    #     mentionByQuery.update(start      =currentMention["start"])
    #     mentionByQuery.update(end        =currentMention["end"])
    #     mentionByQuery.update(boundaryKey=str(currentMention["docId"])+" "+str(currentMention["start"])+" "+str(currentMention["end"]))
    #
    #     mentionByQuery.update(name=currentMention["name"])
    #     mentionByQuery.update(neType=currentMention["goldType"])
    #     mentionByQuery.update(prob=1.0)
    #     mentionByQuery.update(probOfContainsPLO=0.0)
    #     mentionByQuery.update(probOfContainedByPLO=0.0)
    #     mentionByQuery.update(probOfOverlapWithPLO=0.0)
    #
    #     if currentMention["neType"] == currentMention["goldType"]:
    #         global numOfHumanTT
    #         numOfHumanTT += 1
    #     else:
    #         global numOfHumanFT
    #         numOfHumanFT += 1
    #
    #     return mentionByQuery


    def querySimilarMention(self, currentMention, query, queryName):
        mapOfMentionKeysByQueryNames = currentMention["mapOfMentionKeysByQueryNames"]
        mentionKeysByQuery = mapOfMentionKeysByQueryNames[queryName]

        mapOfQueriedMentionSizeByQueries = currentMention["mapOfQueriedMentionSizeByQueries"]
        queriedMentionSize = mapOfQueriedMentionSizeByQueries[query]

        avalible = self.availble(currentMention, query)
        if avalible[0]:
            mentionKeyByQuery = mentionKeysByQuery[queriedMentionSize]
            mentionByQuery = self.mapOfKeyToMention[mentionKeyByQuery]
            mapOfQueriedMentionSizeByQueries[query] = queriedMentionSize + 1
        else:
            mentionByQuery = None
        return mentionByQuery


    def availble(self, currentMention, query):
        queryName = mapOfIntToQueryName[query]

        mapOfMentionKeysByQueryNames = currentMention["mapOfMentionKeysByQueryNames"]
        mentionKeysByQuery = mapOfMentionKeysByQueryNames[queryName]

        mapOfQueriedMentionSizeByQueries = currentMention["mapOfQueriedMentionSizeByQueries"]
        queriedMentionSize = mapOfQueriedMentionSizeByQueries[query]

        avalible = queriedMentionSize < len(mentionKeysByQuery)
        desc = "%d-%d" % (len(mentionKeysByQuery), queriedMentionSize)

        return avalible, desc
    #################################################### queryMention


    #################################################### performAction
    # return assessment of the action
    def performAction(self, action, query, currentMention, mentionByQuery):
        # if action == REJECT:
        #     if mentionByQuery and currentMention["neType"] != currentMention["goldType"] and mentionByQuery["neType"] == currentMention["goldType"]:
        #         # should accept
        #         assessment = "REJECT[-1]"
        #     else:
        #         assessment = "REJECT[0]"

        # todo: this branch should be shorted by the branch of human
        if constants.flagOfPropagation and action == ACCEPT:
            # if mentionByQuery and currentMention["neType"] != currentMention["goldType"] and mentionByQuery["neType"] == currentMention["goldType"]:
            #     assessment = "ACCEPT[1]"
            # elif mentionByQuery and currentMention["neType"] == currentMention["goldType"] and mentionByQuery["neType"] != currentMention["goldType"]:
            #     assessment = "ACCEPT[-1]"
            # else:
            #     assessment = "ACCEPT[0]"
            if mentionByQuery:
                currentMention["neType"] = mentionByQuery["neType"]
                currentMention["prob"] = mentionByQuery["probAsEvi"]

        if (constants.flagOfSelection and query == ASK):
            self.humanMentions.append(currentMention)
            if currentMention["neType"] == currentMention["goldType"]:
                global numOfHumanTT
                numOfHumanTT += 1
                # assessment = "ASK[-1]"
            else:
                global numOfHumanFT
                numOfHumanFT += 1
                # assessment = "ASK[1]"
            currentMention["neType"] = currentMention["goldType"]
            currentMention["prob"] = 1.0
            currentMention["probAsEvi"] = 1.0
            currentMention["hasQueriedHuman"] = True
            if currentMention["neType"] != "O":
                currentMention["probOfContainsPLO"] = 0.0
                currentMention["probOfContainedByPLO"] = 0.0
                currentMention["probOfOverlapWithPLO"] = 0.0
                currentMention["probOfPer"] = 0.0
                currentMention["probOfLoc"] = 0.0
                currentMention["probOfOrg"] = 0.0
                if currentMention["neType"] == "PER":
                    currentMention["probOfPer"] = 1.0
                elif currentMention["neType"] == "LOC":
                    currentMention["probOfLoc"] = 1.0
                else:
                    currentMention["probOfOrg"] = 1.0

        if (constants.flagOfSelection and query == ASK) or query == FINISH:
            self.finishedMentions.append(currentMention)
            self.unfinishedMentions.pop(0)

        # elif query == FINISH:
        #     if mentionByQuery and currentMention["neType"] != currentMention["goldType"] and mentionByQuery["neType"] == currentMention["goldType"]:
        #         # should accept
        #         assessment = "FINISH[-1]"
        #     else:
        #         assessment = "FINISH[0]"

        # return assessment
        return
    #################################################### performAction


    def calculateReward(self, action, query, oldNeType, newNeType, goldType, oriProb, oldProb, newProb, oldHasQueriedHuman):
        oldCorrect = 1.0 if oldNeType==goldType else 0.0
        newCorrect = 1.0 if newNeType==goldType else 0.0
        reward = newCorrect - oldCorrect

        # shaping
        # probChangeOfStep = newProb - oldProb
        # reward += probChangeOfStep
        # self.probChangeOfEpisode += probChangeOfStep

        # if (constants_idner.flagOfAsk and query == ASK) or query == FINISH:
        #     if self.currentMentionIdx == len(self.mentionsInDoc) - 1:
        #         reward -= self.probChangeOfEpisode

        if constants.flagOfSelection and query == ASK:
            reward -= constants.humanCost
        elif query == QUERY_IDENTICAL or query == QUERY_SUPER or query == QUERY_SUB:
            reward -= constants.queryCost
            # if self.mentionByLastQuery is None:
            #     reward -= 10
        elif query == FINISH:
            reward -= constants.finishCost

        if not constants.flagOfSelection:
            if query == ASK:
                reward -= 1

        # if action == ACCEPT:
        #     if oldHasQueriedHuman:
        #         reward -= 2.0

        if constants.flagOfRewardRate:
            reward = reward * self.rateFactor

        # reward -= constants_idner.penalty

        return reward


    # take a single step in the episode
    def step(self, action, query):
        actionName = mapOfIntToActionName[action]
        queryName = mapOfIntToQueryName[query]

        if self.currentMentionIdx>=len(self.mentionsInDoc):
            print env.currentMentionIdx
            print len(env.mentionsInDoc)
            print env.doc["id"]
            raise RuntimeError()
        currentMention = self.mentionsInDoc[self.currentMentionIdx]

        goldType = currentMention["goldType"]
        oldNeType = currentMention["neType"]
        oldProb = currentMention["prob"]
        oldHasQueriedHuman = currentMention["hasQueriedHuman"]

        # performAction
        self.performAction(action, query, currentMention, self.mentionByLastQuery)
        newNeType = currentMention["neType"]
        newProb = currentMention["prob"]

        # calculateReward
        oriProb = currentMention["oriProb"]
        reward = self.calculateReward(action, query, oldNeType, newNeType, goldType, oriProb, oldProb, newProb, oldHasQueriedHuman)

        if evalMode:
            oldCorrectStr = "T" if goldType==oldNeType else "F"
            newCorrectStr = "T" if goldType==newNeType else "F"
            labelTrans = oldCorrectStr+"->"+newCorrectStr
            label_trans_map[labelTrans] += 1

            if DEBUG:
                # should use last query
                stepSummary = \
                    "oldValue=" + oldNeType \
                      + "\nnewValue=" + newNeType \
                      + "\nreward\t=" + format(reward * 1.0, "0.9") \
                      + "\n[" + goldType +"]" \
                      + "\t" + oldNeType \
                      + "\t" + format(currentMention["prob"],"0.3f") \
                      + "\t" + newNeType \
                      + "\t" + queryName \
                      + "\t" + actionName \
                      + "\t" + labelTrans \
                      + "\t" + str(currentMention["docId"]) \
                      + "\t" + str(currentMention["start"]) \
                      + "\t" + str(currentMention["end"]) \
                      + "\t" + currentMention["name"]
                detailOutFile.write("\n"+stepSummary)
                if DEV:
                    print stepSummary

        self.stepSeq += 1

        if query == FINISH or (constants.flagOfSelection and query == ASK):
            self.currentMentionIdx = self.findNextUnannotatedMentionIdx(self.currentMentionIdx)
            if self.currentMentionIdx == self.numOfMentionsInDoc:
                self.terminal = True
                queryForConstructState = None
            else:
                # next mention
                self.stepSeqForCurrentMention = 1
                queryForConstructState = self.selectInitialQuery()
                self.mentionByLastQuery = None
        else:
            self.stepSeqForCurrentMention += 1
            queryForConstructState = query

        if not self.terminal:
            # current action performed
            # use current query to construct next state
            self.constructState(queryForConstructState)

        return self.state, reward, self.terminal


# numOfTestingGold 没有用，仅显示
# numOfTesting     accuracy = (numOfTestingCorrect * 1.0) / numOfTesting
# numOfTestingCorrect
# testingPredPLOKeys  numOfPredPLOs, numOfCorrectPLOs
def evaluateDoc(doc):
    global numOfEvalingCorrect, numOfEvalingGold, numOfEvaling
    # global docLevelErrorRates, docLevelHumanRates

    numOfMentionsInDoc = len(doc["mentions"])
    # noinspection PyUnboundLocalVariable
    numOfEvalingGold += numOfMentionsInDoc
    # noinspection PyUnboundLocalVariable
    numOfEvaling += numOfMentionsInDoc

    numOfTestingCorrectInDoc = 0
    for mention in doc["mentions"]:
        goldType = mention["goldType"]
        neType = mention["neType"]
        if neType == goldType:
            # noinspection PyUnboundLocalVariable
            numOfEvalingCorrect += 1
            numOfTestingCorrectInDoc += 1
        if neType != 'O':
            evalingPredPLOKeys.append(mention["boundaryKey"] + " " + neType)

    # docLevelErrorRate = 1 - (numOfTestingCorrectInDoc*1.0 / numOfMentionsInDoc)
    # docLevelHumanRate = len(self.humanMentions)*1.0 / numOfMentionsInDoc

    # docLevelErrorRates.append(docLevelErrorRate)
    # docLevelHumanRates.append(docLevelHumanRate)

    return


# lutm: NA
# called in testEnd()
def plot_hist(evalconf, name):
    for i in evalconf.keys():
        plt.hist(evalconf[i], bins=[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(name+"_"+str(i)+".png")
        plt.clf()
    return

############################################################## experiment-level operations
def initMention(mention):
    mention.update(oriProb=mention["prob"])
    # when accept similar mention, prob changes while probAsEvi does not
    # when queried as similar mention, prob in state uses probAsEvi
    mention.update(probAsEvi=mention["prob"])

    mention["mapOfQueriedMentionSizeByQueries"] = {}
    mention["mapOfQueriedMentionSizeByQueries"].update({QUERY_IDENTICAL: 0})
    mention["mapOfQueriedMentionSizeByQueries"].update({QUERY_SUPER    : 0})
    mention["mapOfQueriedMentionSizeByQueries"].update({QUERY_SUB      : 0})

    return

def initDoc(doc):
    for mention in doc["mentions"]:
        initMention(mention)
    return

def initDocs(docs):
    for doc in docs:
        initDoc(doc)
    return
############################################################## experiment-level operations

#lutm: OK
def main():  # lutm: Python 中，一个变量的作用域总是由在代码中被赋值的地方所决定的。函数定义的是本地作用域，而模块定义的是全局作用域。
    global evalMode
    global numOfEvalingCorrect, numOfEvalingGold, numOfEvaling, EVALCONF, EVALCONF2
    global numOfHumanTT, numOfHumanFT
    global QUERY, ACTION, CHANGES
    global train_docs
    global  dev_docs,  devingGoldPLOKeys, objectives_dev,  detailOutFile_dev , outFile_dev , outFile2_dev , idxOfEvalEpoch_dev ,  devingPredPLOKeys
    global test_docs, testingGoldPLOKeys, objectives_test, detailOutFile_test, outFile_test, outFile2_test, idxOfEvalEpoch_test, testingPredPLOKeys
    global eval_docs, evalingGoldPLOKeys, objectives     , detailOutFile     , outFile     , outFile2     , idxOfEvalEpoch     , evalingPredPLOKeys
    global label_trans_map
    global DEBUG
    global DEV
    global currentLR
    global numOfLRChanged
    global epochIdxOfReachingMaxObj

    objectives_dev  = []
    idxOfEvalEpoch_dev = 0
    objectives_test = []
    idxOfEvalEpoch_test = 0
    currentLR = constants.lrStart
    numOfLRChanged = 0
    epochIdxOfReachingMaxObj = -1

    DEBUG = True
    DEV = False

    print "loading " + resDir + "train-q.docs.json"
    train_docs  = fileUtil.readToObject(resDir + "train-q.docs.json")
    initDocs(train_docs)
    print "train_docs="+str(len(train_docs))

    print "loading " + resDir + "test.docs.json"
    test_docs  = fileUtil.readToObject(resDir + "test.docs.json")
    initDocs(test_docs)
    print "test_docs="+str(len(test_docs))

    print "loading " + resDir + "test.goldPLOKeys.json"
    testingGoldPLOKeys = fileUtil.readToObject(resDir + "test.goldPLOKeys.json")
    print "testingGoldPLOKeys="+str(len(testingGoldPLOKeys))

    print "loading " + resDir + "dev.docs.json"
    dev_docs  = fileUtil.readToObject(resDir + "dev.docs.json")
    initDocs(dev_docs)
    print "dev_docs="+str(len(dev_docs))

    print "loading " + resDir + "dev.goldPLOKeys.json"
    devingGoldPLOKeys = fileUtil.readToObject(resDir + "dev.goldPLOKeys.json")
    print "devingGoldPLOKeys="+str(len(devingGoldPLOKeys))

    docs = train_docs

    idxOfEpisodeInEpoch = 0

    outFilePath_test = outDir + "run.out"
    if os.path.exists(outFilePath_test):
        print "file exists: "+outFilePath_test
        raise RuntimeError()
    outFile_test = open(outFilePath_test, 'w', 0) #unbuffered
    outFile_test.write("dataset=" + constants.dataset + "\n")
    outFile_test.write("classifierModel=" + constants.classifierModel + "\n")
    outFile_test.write("outDir="+outDir+"\n")
    outFile_test.write("humanCost=" + str(constants.humanCost) + "\n")
    outFile_test.write("penalty=" + str(constants.penalty) + "\n")

    outFilePath_dev = outDir + "run.dev.out"
    outFile_dev = open(outFilePath_dev, 'w', 0) #unbuffered

    outFile2_test = open(outFilePath_test + '.2', 'w', 0) #for analysis

    outFile2_dev = open(outFilePath_dev + '.2', 'w', 0) #for analysis

    detailOutFile = None

    #server setup
    port = constants.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Started server on port", port

    print ""
    print "****************  COMMAND  ********************"
    print "cd ~/dqnner-c-lrs/Code/code/dqn"
    print "./run_cpu.sh" \
            " " + constants.dataset + \
            " SVM_LibSVM_RBF" +\
            " " + str(port) +\
            " " + outDir + \
            " " + str(STATE_SIZE) + \
            " " + str(len(train_docs)) + " " + str(len(test_docs)) + " " + str(len(dev_docs)) + " NER_C 20 20" + \
            " " + str(constants.discount) + \
            " 5 " + \
            " " + str(constants.humanCost) +\
            " 0 0 None 55" +\
            " " + str(constants.lrStart) + " " + str(constants.lrStart) + " " + str(1000000)

        #for analysis
    stepCnt = 0
    trainingStepCnt = 0

    # server loop
    while True:                                     # lutm: 这个设计很有意思哦，可以参考；本地网络，应该速度很快吧
        #  Wait for next request from client
        message = socket.recv()                     # lutm: 接收到的情况包括：newGame, testStart, testEnd, step
        # print "Received request: ", message

        if message == "newGame":
            while True:
                docIdx = idxOfEpisodeInEpoch % len(docs)   # docs可能是train_docs，也可能是test_docs
                idxOfEpisodeInEpoch += 1
                doc = docs[docIdx]
                # if doc["numOfMentionsSelected"]<len(doc["mentions"]):
                if True:
                    break
                else:
                    detailOutFile.write("\n||||||||||| SKIP: docIdx=" + str(docIdx) + "\tepisodes=" + str(idxOfEpisodeInEpoch) + "\tdocId=" + str(doc["id"]) + "\t|M|=" + str(len(doc["mentions"])))
                    evaluateDoc(doc)

            if evalMode and DEBUG:
                detailOutFile.write("\n\n")
                detailOutFile.write("\n***************************************************************")
                detailOutFile.write("\nnewGame: docIdx=" + str(docIdx) + "\tepisodes=" + str(idxOfEpisodeInEpoch) + "\tdocId=" + str(doc["id"]) + "\t|M|=" + str(len(doc["mentions"])))
                detailOutFile.write("\n***************************************************************")
            if DEV:
                print("\nnewGame: docIdx=" + str(docIdx) + "\tepisodes=" + str(idxOfEpisodeInEpoch) + "\tdocId=" + str(doc["id"]))

            env = Environment(doc, docIdx, evalMode)
            newstate, reward, terminal = env.state, 0, 'false'

            isNullByLastQuery = "false" if env.mentionByLastQuery else "true"
            isHumanByLastQuery = "false"
            currentMention = env.mentionsInDoc[env.currentMentionIdx]
            hasQueriedHuman = "true" if currentMention["hasQueriedHuman"] else "false"

            #send message (IMP: only for newGame or step messages)
            # outMsg = 'state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = ' + str(newstate) + ',' + str(reward) + ',' + terminal + ',' + str(env.stepSeqForCurrentMention) + ',' + isNullByLastQuery + ',' + isHumanByLastQuery + ',' + hasQueriedHuman
            outMsg = 'state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, currentLR = ' + str(newstate) + ',' + str(reward) + ',' + terminal + ',' + str(env.stepSeqForCurrentMention) + ',' + isNullByLastQuery + ',' + isHumanByLastQuery + ',' + hasQueriedHuman + "," + format(currentLR, "0.15f")
            socket.send(outMsg.replace('[', '{').replace(']', '}'))

        elif message == "testStart" or message == "devStart":
            evalMode = True
            if message == "devStart":
                devMode = True
                docs = dev_docs
                idxOfEvalEpoch_dev += 1
                idxOfEvalEpoch = idxOfEvalEpoch_dev
                outFile = outFile_dev
                outFile2 = outFile2_dev
                detailDir = detailDir_dev
                evalingGoldPLOKeys = devingGoldPLOKeys
            else:
                devMode = False
                docs = test_docs
                idxOfEvalEpoch_test += 1
                idxOfEvalEpoch = idxOfEvalEpoch_test
                outFile = outFile_test
                outFile2 = outFile2_test
                detailDir = detailDir_test
                evalingGoldPLOKeys = testingGoldPLOKeys

            detailOutFile = open(detailDir + format(idxOfEvalEpoch, "03d") +".txt", 'w')

            print "\n*---- ", message
            print "expName=", constants.expName
            print "humanCost=", constants.humanCost
            print "trainingStepCnt=", trainingStepCnt
            print "LR=", str(currentLR)
            if constants.queryType:
                print "queryType=", constants.queryType
            else:
                print "queryType=ALL"
            if constants.selectionStrategy == "confidence":
                print "preHumanRatio=", constants.preHumanRatio

            outFile.write("trainingStepCnt: " + str(trainingStepCnt) + '\n')
            trainingStepCnt = 0

            label_trans_map = collections.defaultdict(lambda: 0)
            if idxOfEvalEpoch==1 or idxOfEvalEpoch==2 or idxOfEvalEpoch==10 or idxOfEvalEpoch==20 or idxOfEvalEpoch%40==0:
                if constants.selectionStrategy == "confidence":
                    DEBUG = constants.preHumanRatio==0.1 or constants.preHumanRatio==0.8
                else:
                    DEBUG = constants.humanCost == 1.0
            else:
                DEBUG = False

            evalingPredPLOKeys = []
            numOfEvalingCorrect = 0
            numOfEvalingGold = 0
            numOfEvaling = 0
            numOfHumanFT = 0
            numOfHumanTT = 0
            QUERY = collections.defaultdict(lambda:0.)
            ACTION = collections.defaultdict(lambda:0.)
            CHANGES = 0
            idxOfEpisodeInEpoch = 0                # 开始评估，清零
            stepCnt = 0

        elif message == "testEnd" or message == "devEnd":
            print "\n---- "
            temp = "\n" + format(idxOfEvalEpoch, ">4") + "\t" + format(idxOfEpisodeInEpoch, ">4") + " ------------\nEvaluation Stats: (Accuracy, Precision, Recall, F1, Doc-H, Doc-E):"
            print temp
            outFile.write(temp+"\n")
            numOfCorrectPLOs = len(set(evalingPredPLOKeys).intersection(set(evalingGoldPLOKeys)))
            numOfPredPLOs = len(evalingPredPLOKeys)
            numOfGoldPLOs = len(evalingGoldPLOKeys)

            accuracy = (numOfEvalingCorrect * 1.0) / numOfEvaling
            prec = (numOfCorrectPLOs * 1.0) / numOfPredPLOs
            rec = (numOfCorrectPLOs * 1.0) / numOfGoldPLOs
            f1 = (2 * prec * rec) / (prec + rec)
            docLevelHumanRate = np.mean(docLevelHumanRates)
            docLevelErrorRate = np.mean(docLevelErrorRates)
            humanRate = (numOfHumanFT+numOfHumanTT) * 1.0 / numOfEvaling

            pScore = label_trans_map["F->T"] - label_trans_map["T->F"]
            if constants.flagOfSelection:
                cScore = numOfHumanTT + numOfHumanFT
                score = pScore - cScore * constants.humanCost
                objective = accuracy - constants.humanCost * humanRate
            else:
                score = pScore
                objective = accuracy

            print                    format(accuracy, "0.4"), format(prec, "0.4"), format(rec, "0.4"), format(f1, "0.4"), format(docLevelHumanRate, "0.4"), format(docLevelErrorRate, "0.4"), "########[CorrectPLOOs", numOfEvalingCorrect, "TotalPLOOs", numOfEvaling, numOfEvalingGold, "] CorrectPLOs", numOfCorrectPLOs, "PredPLOs", numOfPredPLOs, "GoldPLOs", len(testingGoldPLOKeys)
            outFile.write('\t'.join([format(accuracy, "0.4"), format(prec, "0.4"), format(rec, "0.4"), format(f1, "0.4"), format(docLevelHumanRate, "0.4"), format(docLevelErrorRate, "0.4")]) + '\n')

            if constants.lrVersion== "2":
                if devMode:
                    if epochIdxOfReachingMaxObj>0:
                        # have reached max objective
                        if idxOfEvalEpoch == epochIdxOfReachingMaxObj + constants.numOfEpochsAfterLRDec:
                            # decrease LR again
                            currentLR *= 0.1
                            numOfLRChanged += 1
                            strOfLRInfo = "\n********************************************************* decrease LR again, LR="+str(currentLR)
                            print strOfLRInfo
                            outFile.write(strOfLRInfo + '\n')
                        elif idxOfEvalEpoch == epochIdxOfReachingMaxObj + constants.numOfEpochsAfterLRDec * 2:
                            # early stop
                            strOfLRInfo = "\n\n********************************************************* early stop, LR="+str(currentLR)
                            print strOfLRInfo
                            outFile.write(strOfLRInfo + '\n')

                    elif objective>=maxObjectiveInLRStudy:
                        currentLR *= 0.1
                        numOfLRChanged += 1
                        epochIdxOfReachingMaxObj = idxOfEvalEpoch
                        strOfLRInfo = "\n********************************************************* reached maxObjectiveInLRStudy " + str(maxObjectiveInLRStudy) + "<=" + str(objective) + ", LR="+str(currentLR)
                        print strOfLRInfo
                        outFile.write(strOfLRInfo + '\n')



            print "StepCnt (total, average):", stepCnt, float(stepCnt)/len(docs)
            outFile.write("StepCnt (total, average): " + str(stepCnt)+ ' ' + str(float(stepCnt)/len(docs)) + '\n')
            labelTrans = \
                " T->T\tF->F\tT->F\tF->T\tT->T(H)\tF->T(H)\t     H\t Score\tObjective\tLR\n" \
                  + format(label_trans_map["T->T"], ">4") \
                  +"\t"+ format(label_trans_map["F->F"], ">4") \
                  +"\t"+ format(label_trans_map["T->F"], ">4") \
                  +"\t"+ format(label_trans_map["F->T"], ">4") \
                  +"\t"+ format(numOfHumanTT, ">6") \
                  +"\t"+ format(numOfHumanFT, ">6") \
                  +"\t"+ format(numOfHumanTT+numOfHumanFT, ">6") \
                  +"\t"+ format(score, ">6") \
                  +"\t"+ format(format(objective, "0.6"), ">6") \
                  +("\t****["+format(maxObjectiveInLRStudy, "0.6")+"]" if devMode and constants.lrVersion=="2" else "") \
                  +"\t"+ str(currentLR)
            print labelTrans
            outFile.write(labelTrans + '\n')

            temp = "\nCost, Accuracy, Objctive, LR"+":\t"+format(humanRate, "0.4")+"\t"+format(accuracy, "0.4")+"\t"+format(objective, "0.6")+"\t"+str(currentLR)
            print temp
            outFile.write(temp+"\n")

            qsum = sum(QUERY.values())
            asum = sum(ACTION.values())
            outFile2.write("------------\nQsum: " + str(qsum) +  " Asum: " +  str(asum)+'\n')
            for k, val in QUERY.items():
                outFile2.write("Query " + str(k) + ' ' + str(val/qsum)+'\n')
            for k, val in ACTION.items():
                outFile2.write("Action " + str(k) + ' ' + str(val/asum)+'\n')
            outFile2.write("CHANGES: "+str(CHANGES)+ ' ' + str(float(CHANGES)/len(docs))+"\n")
            outFile2.write("STAT_POSITIVE, STAT_NEGATIVE "+str(STAT_POSITIVE) + ', ' +str(STAT_NEGATIVE)+'\n')

            evalMode = False
            idxOfEpisodeInEpoch = 0    # 评估结束
            docs = train_docs

            print "\n**** ", message

            if not devMode: # testEnd
                if epochIdxOfReachingMaxObj > 0:
                    # have reached max objective
                    if idxOfEvalEpoch == epochIdxOfReachingMaxObj + constants.numOfEpochsAfterLRDec:
                        # early stop
                        strOfLRInfo = "********************************************************* earlyStop, LR=" + str(currentLR)
                        print strOfLRInfo
                        outFile.write(strOfLRInfo + '\n')
                        socket.send("earlyStop")  # lutm: 给agent发
                        break
                print "\n------------------------------------------------  Next epoch ", (idxOfEvalEpoch+1)

        else:
            # message is "step"
            stepCnt += 1
            if not evalMode:
                trainingStepCnt += 1

            action, query = [int(q) for q in message.split()]
            actionName = mapOfIntToActionName[action]
            queryName = mapOfIntToQueryName[query]
            # if env.currentMentionIdx>=len(env.mentionsInDoc):
            #     print env.currentMentionIdx
            #     print len(env.mentionsInDoc)
            #     print env.doc["id"]
            #     raise RuntimeError()
            # currentMention = env.mentionsInDoc[env.currentMentionIdx]
            # if DEV:
            #     print("STEP ["+str(env.stepSeq)+"]"+currentMention["boundaryKey"]+"\t["+str(env.currentMentionIdx)+", "+str(env.stepSeqForCurrentMention)+"]\t"+currentMention["name"])
            #     print(actionName + "\t" + queryName)
            # if action == 0:
            #    print("message=["+message+"]")

            # forced action
            # noinspection PyUnboundLocalVariable
            # if env.stepSeqForCurrentMention >= 10 or currentMention["hasQueriedHuman"]:
            # if env.stepSeqForCurrentMention >= 10:
            #     action = FINISH
            #     actionName = mapOfIntToActionName[action]
            #     if DEV:
            #         print(actionName + "\t" + queryName)
            # elif env.mentionByLastQuery == None:
            #     if action==ACCEPT:
            #         action = REJECT
            #         actionName = mapOfIntToActionName[action]
            #         if DEV:
            #             print(actionName + "\t" + queryName)
            # elif env.mentionByLastQuery["isHumanMention"]:
            #     action = ACCEPT
            #     actionName = mapOfIntToActionName[action]
            #     if DEV:
            #         print(actionName + "\t" + queryName)

            if evalMode and DEBUG:
                detailOutFile.write("\nACTION=" + str(action) + " " + actionName + "\nQUERY=" + str(query) + " " + queryName)
                # queryName = constants_idner.mapOfIntToQueryName[queryIdx]
                # detailOutFile.write("query\t=" + format(queryIdx, ">3") + "\t" + queryName + "\n")

            if evalMode:
                ACTION[action] += 1   # lutm: 看样子是个计数器
                QUERY[query] += 1

            # noinspection PyUnboundLocalVariable
            newstate, reward, terminal = env.step(action, query)

            env.lastAction = action
            env.lastQuery = query

            terminal = 'true' if terminal else 'false'
            if evalMode and DEBUG:
                detailOutFile.write("\nTERMINAL\t=\t" + str(terminal))

            if not env.terminal:
                isNullByLastQuery = "false" if env.mentionByLastQuery else "true"
                isHumanByLastQuery = "false"
                currentMention = env.mentionsInDoc[env.currentMentionIdx]
                hasQueriedHuman = "true" if currentMention["hasQueriedHuman"] else "false"
            else:
                isNullByLastQuery = "true"
                isHumanByLastQuery = "false"
                hasQueriedHuman = "false"

            #send message (IMP: only for newGame or step messages)
            outMsg = 'state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = ' + str(newstate) + ',' + str(reward) + ',' + terminal + ',' + str(env.stepSeqForCurrentMention) + ',' + isNullByLastQuery + ',' + isHumanByLastQuery + ',' + hasQueriedHuman
            socket.send(outMsg.replace('[', '{').replace(']', '}'))

        if message != "testStart" and message != "testEnd" and message != "devStart" and message != "devEnd":
            # do doc eval if terminal
            # evalMode=True，表示当前在testStart和testEnd之间，所以需要评估
            # docs是测试集中的所有文章，docNum是已经评估的文章，这个意思是：评估一遍所有的文章
            # 现在的设计是，每一步都有结果，所以每一步都可以存储结果（如果是评估模式）
            # 如果有的步子不能出结果，那么这里要加上一个条件，例如“当前动作是出结果的”
            # 等到一篇文章（一个情节）结束（terminal== true'），是一个方法。
            # noinspection PyUnboundLocalVariable
            if evalMode and idxOfEpisodeInEpoch <= len(docs) and terminal=='true':
                # 一篇文章经过多次精化、查询结束了，才进入评估
                evaluateDoc(env.doc)
        else:
            socket.send("done")                         # lutm: 给agent发


def mentionToStr(mention):
    if mention == None:
        return "None"
    else:
        if mention["hasQueriedHuman"]:
            temp = "H[1]"
        else:
            temp = "H[0]"
        return temp + "\t" + mention["boundaryKey"] \
           + "\tGold: " + mention["goldType"] \
           + "\tPred: " + mention["neType"] \
           + "\t" + format(mention["prob"], "0.3") \
           + "\t" + mention["prevText"].replace("\n", "[\\n]") + " ||||" \
           + " " + mention["name"].replace("\n", "[\\n]") \
           + " |||| " + mention["succText"].replace("\n", "[\\n]") + ""

    # + " [" + mention["prevPosTag1"] + "]" \
           # + " [" + mention["startPosTag"] + "]" \
           # + " [" + mention["endPosTag"] + "]" \
           # + " [" + mention["succPosTag1"] + "]" \
           # + " [" + originaldoc["prevPosTag2"] + "]" \
           # + " [" + originaldoc["succPosTag2"] + "]" \


def strLastNObjectives():
    startIdx = len(objectives) - constants.lrEpochsCheck
    if startIdx<0:
        startIdx=0
        str = ""
    else:
        str = format(objectives[0], "0.6")
    for i in range(startIdx, len(objectives)):
        if i == startIdx:
            str += "; "
        else:
            str += ", "
        str += format(objectives[i], "0.6")
    return str


def hasObjectiveStayedOrDecreased(currentObjective):
    length = len(objectives)
    if length<constants.lrEpochsCheck:
        return False

    lastNthObjective = objectives[length - constants.lrEpochsCheck]
    diff = currentObjective-lastNthObjective
    if diff<0.000001:
        return True
    else:
        return False


if __name__ == '__main__':                              # lutm: 当运行这个文件时，这里是入口
    env = None                                          # lutm: None相当于Java的null
    newstate, reward, terminal = None, None, None       # lutm：这样赋值也不错

main()                                          # lutm: 调用main函数

#sample
#python server.py --port 7000 --trainEntities consolidated/train+context.5.p --testEntities consolidated/dev+test+context.5.p --outFile outputs/tmp2.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2


