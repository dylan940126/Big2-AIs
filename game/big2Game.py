#big 2 class
#import enumerateOptions
#import gameLogic
from . import enumerateOptions
from . import gameLogic

import numpy as np
import random
import math

def convertAvailableActions(availAcs):
    #convert from (1,0,0,1,1...) to (0, -math.inf, -math.inf, 0,0...) etc
    availAcs[np.nonzero(availAcs==0)] = -math.inf
    availAcs[np.nonzero(availAcs==1)] = 0
    return availAcs

class handPlayed:
    def __init__(self, hand, player):
        self.hand = hand
        self.player = player
        self.nCards = len(hand)
        if self.nCards <= 3:
            self.type = 1
        elif self.nCards == 4:
            if gameLogic.isFourOfAKind(hand):
                self.type = 2
            else:
                self.type = 1
        elif self.nCards == 5:
            if gameLogic.isStraight(hand):
                if gameLogic.isFlush(hand):
                    self.type = 4
                else:
                    self.type = 1
            elif gameLogic.isFlush(hand):
                self.type = 2
            else:
                self.type = 3

class big2Game:
    def __init__(self):
        self.reset()
        
    def reset(self):
        shuffledDeck = np.random.permutation(52) + 1
        #hand out cards to each player
        self.currentHands = {}
        self.currentHands[1] = np.sort(shuffledDeck[0:13])
        self.currentHands[2] = np.sort(shuffledDeck[13:26])
        self.currentHands[3] = np.sort(shuffledDeck[26:39])
        self.currentHands[4] = np.sort(shuffledDeck[39:52])
        self.cardsPlayed = np.zeros((4,52), dtype=int)
        #找出擁有梅花三(3C)的玩家 - 這張牌會被首先打出
        threeClubInd = np.where(shuffledDeck == 1)[0][0]  # 梅花三的編號是1
        if threeClubInd < 13:
            whoHas3C = 1
        elif threeClubInd < 26:
            whoHas3C = 2
        elif threeClubInd < 39:
            whoHas3C = 3
        else:
            whoHas3C = 4
        self.currentHands[whoHas3C] = self.currentHands[whoHas3C][1:]
        self.cardsPlayed[whoHas3C-1][0] = 1  # 梅花三(3C)的索引是0（1-1）
        self.goIndex = 1
        self.handsPlayed = {}
        self.handsPlayed[self.goIndex] = handPlayed([1], whoHas3C)  # 梅花三(3C)的編號是1
        self.goIndex += 1
        self.playersGo = whoHas3C + 1
        if self.playersGo == 5:
            self.playersGo = 1
        self.passCount = 0
        self.control = 0
        self.gameOver = 0
        self.rewards = np.zeros((4,))
        self.goCounter = 0
    
    def updateGame(self, option, nCards=0):
        self.goCounter += 1
        if option == -1:
            #they pass
            cPlayer = self.playersGo
            self.playersGo += 1
            if self.playersGo == 5:
                self.playersGo = 1
            self.passCount += 1
            if self.passCount == 3:
                self.control = 1
                self.passCount = 0
            return
        self.passCount = 0
        if nCards == 1:
            handToPlay = np.array([self.currentHands[self.playersGo][option]])
        elif nCards == 2:
            handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseTwoCardIndices[option]]
        elif nCards == 3:
            handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseThreeCardIndices[option]]
        elif nCards == 4:
            handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseFourCardIndices[option]]
        else:
            handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseFiveCardIndices[option]]
        for i in handToPlay:
            self.cardsPlayed[self.playersGo-1][i-1] = 1
        self.handsPlayed[self.goIndex] = handPlayed(handToPlay, self.playersGo)
        self.control = 0
        self.goIndex += 1
        self.currentHands[self.playersGo] = np.setdiff1d(self.currentHands[self.playersGo],handToPlay)
        if self.currentHands[self.playersGo].size == 0:
            self.assignRewards()
            self.gameOver = 1
            return
        self.playersGo += 1
        if self.playersGo == 5:
            self.playersGo = 1
            
    def assignRewards(self):
        totCardsLeft = 0
        for i in range(1,5):
            nC = self.currentHands[i].size
            if nC == 0:
                winner = i
            else:
                self.rewards[i-1] = -1*nC
                totCardsLeft += nC
        self.rewards[winner-1] = totCardsLeft
        
    def randomOption(self):
        cHand = self.currentHands[self.playersGo]
        if self.control == 0:
            prevHand = self.handsPlayed[self.goIndex-1].hand
            nCards = len(prevHand)
            if nCards > 1:
                handOptions = gameLogic.handsAvailable(cHand)
            if nCards == 1:
                options = enumerateOptions.oneCardOptions(cHand,prevHand,1)
            elif nCards == 2:
                options = enumerateOptions.twoCardOptions(handOptions, prevHand, 1)
            elif nCards == 3:
                options = enumerateOptions.threeCardOptions(handOptions, prevHand, 1)
            elif nCards == 4:
                if gameLogic.isFourOfAKind(prevHand):
                    options = enumerateOptions.fourCardOptions(handOptions, prevHand, 2)
                else:
                    options = enumerateOptions.fourCardOptions(handOptions, prevHand, 1)
            else:
                if gameLogic.isStraight(prevHand):
                    if gameLogic.isFlush(prevHand):
                        options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 4)
                    else:
                        options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 1)
                elif gameLogic.isFlush(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 2)
                else:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 3)
            if isinstance(options,int):
                nOptions = -1
            else:
                nOptions = len(options)
            ind = random.randint(0,nOptions)
            if ind == nOptions or isinstance(options,int):
                return -1 #pass
            else:
                return (options[ind], nCards)
        else:
            #we have control - choose from any option
            handOptions = gameLogic.handsAvailable(cHand)
            oneCardOptions = enumerateOptions.oneCardOptions(cHand)
            twoCardOptions = enumerateOptions.twoCardOptions(handOptions)
            threeCardOptions = enumerateOptions.threeCardOptions(handOptions)
            fourCardOptions = enumerateOptions.fourCardOptions(handOptions)
            fiveCardOptions = enumerateOptions.fiveCardOptions(handOptions)
            if isinstance(oneCardOptions, int):
                n1 = 0
            else:
                n1 = len(oneCardOptions)
            if isinstance(twoCardOptions, int):
                n2 = 0
            else:
                n2 = len(twoCardOptions)
            if isinstance(threeCardOptions, int):
                n3 = 0
            else:
                n3 = len(threeCardOptions)
            if isinstance(fourCardOptions, int):
                n4 = 0
            else:
                n4 = len(fourCardOptions)
            if isinstance(fiveCardOptions, int):
                n5 = 0
            else:
                n5 = len(fiveCardOptions)
            nTot = n1 + n2 + n3 + n4 + n5
            ind = random.randint(0,nTot-1)
            if ind < n1:
                return (oneCardOptions[ind],1)
            elif ind < (n1+n2):
                return (twoCardOptions[ind-n1],2)
            elif ind < (n1+n2+n3):
                return (threeCardOptions[ind-n1-n2],3)
            elif ind < (n1+n2+n3+n4):
                return (fourCardOptions[ind-n1-n2-n3],4)
            else:
                return (fiveCardOptions[ind-n1-n2-n3-n4],5)
            
    def returnAvailableActions(self):
    
        currHand = self.currentHands[self.playersGo]
        availableActions = np.zeros((enumerateOptions.nActions[5]+1,))
        
        if self.control == 0:
            #allow pass action
            availableActions[enumerateOptions.passInd] = 1
            
            prevHand = self.handsPlayed[self.goIndex-1].hand
            nCardsToBeat = len(prevHand)
            
            if nCardsToBeat > 1:
                handOptions = gameLogic.handsAvailable(currHand)
                
            if nCardsToBeat == 1:
                options = enumerateOptions.oneCardOptions(currHand, prevHand,1)
            elif nCardsToBeat == 2:
                options = enumerateOptions.twoCardOptions(handOptions, prevHand, 1)
            elif nCardsToBeat == 3:
                options = enumerateOptions.threeCardOptions(handOptions, prevHand, 1)
            elif nCardsToBeat == 4:
                if gameLogic.isFourOfAKind(prevHand):
                    options = enumerateOptions.fourCardOptions(handOptions, prevHand, 2)
                else:
                    options = enumerateOptions.fourCardOptions(handOptions, prevHand, 1)
            else:
                if gameLogic.isStraight(prevHand):
                    if gameLogic.isFlush(prevHand):
                        options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 4)
                    else:
                        options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 1)
                elif gameLogic.isFlush(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 2)
                else:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 3)
                    
            if isinstance(options, int): #no options - must pass
                return availableActions
            
            for option in options:
                index = enumerateOptions.getIndex(option, nCardsToBeat)
                availableActions[index] = 1
                
            return availableActions
        
        
        else: #player has control.
            handOptions = gameLogic.handsAvailable(currHand)
            oneCardOptions = enumerateOptions.oneCardOptions(currHand)
            twoCardOptions = enumerateOptions.twoCardOptions(handOptions)
            threeCardOptions = enumerateOptions.threeCardOptions(handOptions)
            fourCardOptions = enumerateOptions.fourCardOptions(handOptions)
            fiveCardOptions = enumerateOptions.fiveCardOptions(handOptions)
            
            for option in oneCardOptions:
                index = enumerateOptions.getIndex(option, 1)
                availableActions[index] = 1
                
            if not isinstance(twoCardOptions, int):
                for option in twoCardOptions:
                    index = enumerateOptions.getIndex(option, 2)
                    availableActions[index] = 1
                    
            if not isinstance(threeCardOptions, int):
                for option in threeCardOptions:
                    index = enumerateOptions.getIndex(option, 3)
                    availableActions[index] = 1
                    
            if not isinstance(fourCardOptions, int):
                for option in fourCardOptions:
                    index = enumerateOptions.getIndex(option, 4)
                    availableActions[index] = 1
                    
            if not isinstance(fiveCardOptions, int):
                for option in fiveCardOptions:
                    index = enumerateOptions.getIndex(option, 5)
                    availableActions[index] = 1
                    
            return availableActions

    def step(self, action):
        opt, nC = enumerateOptions.getOptionNC(action)
        self.updateGame(opt, nC)
        if self.gameOver == 0:
            reward = 0
            done = False
            info = None
        else:
            reward = self.rewards
            done = True
            info = {}
            info['numTurns'] = self.goCounter
            info['rewards'] = self.rewards
        return reward, done, info
    
    def getCurrentState(self):
        return self.playersGo, None, convertAvailableActions(self.returnAvailableActions()).reshape(1,1695)
        
    def getInfoForDrawing(self):
        prevHands = []
        gInd = self.goIndex-1
        if gInd == 1:
            sInd = gInd
            fInd = gInd + 1
        elif gInd == 2:
            sInd = gInd - 1
            fInd = gInd + 1
        else:
            sInd = gInd - 2
            fInd = gInd + 1
        for i in range(sInd,fInd):
            toAppend = self.handsPlayed[i].hand
            if not isinstance(toAppend, list):
                toAppend = toAppend.tolist()
            prevHands.append(toAppend)
        while len(prevHands) < 3:
            prevHands.append([])
        info = {
            'type' : "updateGame",
            'playersHand' : self.currentHands[1].tolist(),
            'playersGo' : self.playersGo,
            'control' : self.control,
            'nCards' : [len(self.currentHands[2]), len(self.currentHands[3]), len(self.currentHands[4])],
            'previousHands' : prevHands,
            'gameOver' : self.gameOver,
            'rewards' : self.rewards.tolist()
        }
        return info