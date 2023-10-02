#enumerate options
import numpy as np
from . import gameLogic

nActions = np.array([13,33,31,330,1287,1694])
nAcSum = np.cumsum(nActions[:-1])

nActions2 = 33
twoCardIndices = np.zeros((13,13), dtype=int)
inverseTwoCardIndices = np.zeros((nActions2,2),dtype=int)
c=0
for c1 in range(12):
    nextInd = np.min([c1+3,12])
    for c2 in range(c1+1,nextInd+1):
        twoCardIndices[c1][c2] = c
        inverseTwoCardIndices[c][0] = c1
        inverseTwoCardIndices[c][1] = c2
        c += 1
nActions3 = 31
threeCardIndices = np.zeros((13,13,13), dtype=int)
inverseThreeCardIndices = np.zeros((nActions3,3), dtype=int)
c=0
for c1 in range(11):
    nextInd = np.min([c1+2,11])
    for c2 in range(c1+1,nextInd+1):
        finalInd = np.min([c1+3,12])
        for c3 in range(c2+1,finalInd+1):
            threeCardIndices[c1][c2][c3] = c
            inverseThreeCardIndices[c][0] = c1
            inverseThreeCardIndices[c][1] = c2
            inverseThreeCardIndices[c][2] = c3
            c += 1
nActions4 = 330
fourCardIndices = np.zeros((13,13,13,13), dtype=int)
inverseFourCardIndices = np.zeros((nActions4,4), dtype=int)
c=0
for c1 in range(10):
    nextInd = np.min([c1+3,10])
    for c2 in range(c1+1,nextInd+1):
        for c3 in range(c2+1,12):
            nextInd = np.min([c3+3,12])
            for c4 in range(c3+1,nextInd+1):
                fourCardIndices[c1][c2][c3][c4] = c
                inverseFourCardIndices[c][0] = c1
                inverseFourCardIndices[c][1] = c2
                inverseFourCardIndices[c][2] = c3
                inverseFourCardIndices[c][3] = c4
                c += 1
nActions5 = 1287 #number of potentially possible 5 card moves
fiveCardIndices = np.zeros((13,13,13,13,13),dtype=int)
inverseFiveCardIndices = np.zeros((nActions5,5),dtype=int)
c = 0
for c1 in range(9):
    for c2 in range(c1+1,10):
        for c3 in range(c2+1,11):
            for c4 in range(c3+1,12):
                for c5 in range(c4+1,13):
                    fiveCardIndices[c1][c2][c3][c4][c5] = c
                    inverseFiveCardIndices[c][0] = c1
                    inverseFiveCardIndices[c][1] = c2
                    inverseFiveCardIndices[c][2] = c3
                    inverseFiveCardIndices[c][3] = c4
                    inverseFiveCardIndices[c][4] = c5
                    c += 1

passInd = nActions[-1]

def getIndex(option, nCards):
    if nCards==0: #pass
        return passInd
    sInd = 0
    for i in range(nCards-1):
        sInd += nActions[i]
    return sInd + option

def getOptionNC(ind):
    if ind == passInd:
        return -1, 0
    if ind < nAcSum[0]:
        return ind, 1
    elif ind < nAcSum[1]:
        return ind - nAcSum[0], 2
    elif ind < nAcSum[2]:
        return ind - nAcSum[1], 3
    elif ind < nAcSum[3]:
        return ind - nAcSum[2], 4
    else:
        return ind - nAcSum[3], 5

def fiveCardOptions(handOptions, prevHand=[],prevType=0):
    #prevType = 0 - no hand, you have control and can play any 5 card
    #         = 1 - straight
    #         = 2 - flush
    #         = 3 - full house
    #         = 4 - straight flush
        
    validInds = np.zeros((nActions[4],),dtype=int)
    c = 0
    cardInds = np.zeros((5,),dtype=int) #reuse
    
    #first deal with straights
    if prevType == 2 or prevType == 3 or prevType == 4:
        pass
    else:
        if len(handOptions.straights) > 0:
            for straight in handOptions.straights:
                nC = straight.size
                for i1 in range(nC-4): #first index of hand
                    val1 = handOptions.cards[straight[i1]].value
                    cardInds[0] = handOptions.cards[straight[i1]].indexInHand
                    for i2 in range(i1+1,nC-3):
                        val2 = handOptions.cards[straight[i2]].value
                        if val1 == val2:
                            continue
                        if val2 > val1 + 1:
                            break
                        cardInds[1] = handOptions.cards[straight[i2]].indexInHand
                        for i3 in range(i2+1,nC-2):
                            val3 = handOptions.cards[straight[i3]].value
                            if val3 == val2:
                                continue
                            if val3 > val2 + 1:
                                break
                            cardInds[2] = handOptions.cards[straight[i3]].indexInHand
                            for i4 in range(i3+1,nC-1):
                                val4 = handOptions.cards[straight[i4]].value
                                if val4 == val3:
                                    continue
                                if val4 > val3 + 1:
                                    break
                                cardInds[3] = handOptions.cards[straight[i4]].indexInHand
                                for i5 in range(i4+1, nC):
                                    val5 = handOptions.cards[straight[i5]].value
                                    if val5 == val4:
                                        continue
                                    if val5 > val4 + 1:
                                        break
                                    #import pdb; pdb.set_trace()
                                    cardInds[4] = handOptions.cards[straight[i5]].indexInHand
                                    if prevType == 1:
                                        if handOptions.cHand[cardInds[4]] < prevHand[4]:
                                            continue
                                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                    c += 1
    
    #now deal with flushes (easier)
    if prevType == 4:
        pass
    else:
        if len(handOptions.flushes) > 0:
            for flush in handOptions.flushes:
                #all combinations of flush are allowable
                nC = flush.size
                for i1 in range(nC-4):
                    cardInds[0] = handOptions.cards[flush[i1]].indexInHand
                    for i2 in range(i1+1,nC-3):
                        cardInds[1] = handOptions.cards[flush[i2]].indexInHand
                        for i3 in range(i2+1,nC-2):
                            cardInds[2] = handOptions.cards[flush[i3]].indexInHand
                            for i4 in range(i3+1,nC-1):
                                cardInds[3] = handOptions.cards[flush[i4]].indexInHand
                                for i5 in range(i4+1,nC):
                                    cardInds[4] = handOptions.cards[flush[i5]].indexInHand
                                    if prevType == 2:
                                        if prevHand[4] > handOptions.cHand[cardInds[4]]:
                                            handBeingPlayed = handOptions.cHand[cardInds]
                                            if gameLogic.isStraight(handBeingPlayed): #its a straight flush so wins
                                                pass
                                            else:
                                                continue
                                    if prevType == 3: #needs to be a straight flush to beat
                                        handBeingPlayed = handOptions.cHand[cardInds]
                                        if gameLogic.isStraight(handBeingPlayed): #its a straight flush so wins
                                            pass
                                        else:
                                            continue
                                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                    c += 1
    
    #now deal with full houses
    if prevType == 4:
        pass
    else:
        if prevType == 3:
            threeVal = gameLogic.cardValue(prevHand[2])
        nPairs = handOptions.nPairs
        nThree = handOptions.nThreeOfAKinds
        if nPairs > 0 and nThree > 0:
            for pair in handOptions.pairs:
                pVal = handOptions.cards[pair[0]].value
                for three in handOptions.threeOfAKinds:
                    tVal = handOptions.cards[three[0]].value
                    if tVal == pVal:
                        continue
                    if pVal > tVal:
                        cardInds[0] = handOptions.cards[three[0]].indexInHand
                        cardInds[1] = handOptions.cards[three[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[2]].indexInHand
                        cardInds[3] = handOptions.cards[pair[0]].indexInHand
                        cardInds[4] = handOptions.cards[pair[1]].indexInHand
                    else:
                        cardInds[0] = handOptions.cards[pair[0]].indexInHand
                        cardInds[1] = handOptions.cards[pair[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[0]].indexInHand
                        cardInds[3] = handOptions.cards[three[1]].indexInHand
                        cardInds[4] = handOptions.cards[three[2]].indexInHand
                    if prevType == 3:
                        if threeVal > tVal:
                            continue
                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                    c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1

    

def fourCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType: 1 - pair, 2 - fourofakind    
    validInds = np.zeros((nActions[3],),dtype=int)
    c = 0
    cardInds = np.zeros((4,),dtype=int) #reuse
    
    #four of a kinds
    if len(handOptions.fourOfAKinds) > 0:
        for four in handOptions.fourOfAKinds:
            cardInds[0] = handOptions.cards[four[0]].indexInHand
            cardInds[1] = handOptions.cards[four[1]].indexInHand
            cardInds[2] = handOptions.cards[four[2]].indexInHand
            cardInds[3] = handOptions.cards[four[3]].indexInHand
            if prevType == 2:
                if handOptions.cHand[cardInds[0]] < prevHand[3]:
                    continue
            validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
            c += 1
    #two pairs
    if prevType == 2:
        pass
    else:
        if handOptions.nDistinctPairs >= 2:
            nPairs = handOptions.nPairs
            for p1 in range(nPairs-1):
                p1Val = handOptions.cards[handOptions.pairs[p1][0]].value
                for p2 in range(p1+1,nPairs):
                    p2Val = handOptions.cards[handOptions.pairs[p2][0]].value
                    if p1Val == p2Val:
                        continue
                    cardInds[0] = handOptions.cards[handOptions.pairs[p1][0]].indexInHand
                    cardInds[1] = handOptions.cards[handOptions.pairs[p1][1]].indexInHand
                    cardInds[2] = handOptions.cards[handOptions.pairs[p2][0]].indexInHand
                    cardInds[3] = handOptions.cards[handOptions.pairs[p2][1]].indexInHand
                    if prevType == 1:
                        if handOptions.cHand[cardInds[3]] < prevHand[3]:
                            continue
                    validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
                    c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1 #no four card hands

def threeCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played three of a kind
    validInds = np.zeros((nActions[2],), dtype=int)
    c = 0
    cardInds = np.zeros((3,), dtype=int) #reuse
    
    if handOptions.nThreeOfAKinds > 0:
        for three in handOptions.threeOfAKinds:
            cardInds[0] = handOptions.cards[three[0]].indexInHand
            cardInds[1] = handOptions.cards[three[1]].indexInHand
            cardInds[2] = handOptions.cards[three[2]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[2]] < prevHand[2]:
                    continue
            validInds[c] = threeCardIndices[cardInds[0]][cardInds[1]][cardInds[2]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1

def twoCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played a pair
    validInds = np.zeros((nActions[1],), dtype=int)
    c = 0
    cardInds = np.zeros((2,), dtype=int)
    
    if handOptions.nPairs > 0:
        for pair in handOptions.pairs:
            cardInds[0] = handOptions.cards[pair[0]].indexInHand
            cardInds[1] = handOptions.cards[pair[1]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[1]] < prevHand[1]:
                    continue
            validInds[c] = twoCardIndices[cardInds[0]][cardInds[1]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1
    
def oneCardOptions(hand, prevHand = [], prevType = 0):
    nCards = len(hand)
    validInds = np.zeros((nCards,), dtype=int)
    c = 0
    for i in range(nCards):
        if prevType == 1:
            if prevHand > hand[i]:
                continue
        validInds[c] = i
        c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1