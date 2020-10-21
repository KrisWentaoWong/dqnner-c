# -*- coding: utf-8 -*-

# 构建从key到mention的映射
def constructMapOfKeyToMention(mentions):
    mapOfKeyToMention = {}
    for mention in mentions:
        key = mention["boundaryKey"]
        mapOfKeyToMention[key] = mention
    return mapOfKeyToMention


def getProbOfPLO(mention):
    return mention["probOfPer"] + mention["probOfLoc"] + mention["probOfOrg"]


def overlap(cdSpan0, cdSpan1):
    if cdSpan0["docId"] == cdSpan1["docId"]:
        start0 = cdSpan0["start"]
        start1 = cdSpan1["start"]
        end0 = cdSpan0["end"]
        end1 = cdSpan1["end"]
        if start0 < start1:
            if end0 > start1:
                return True
            else:
                return False
        else:
            if start0 == start1:
                return True
            else:
                if start0 < end1:
                    return True
                else:
                    return False
    else:
        return False
