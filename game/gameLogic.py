import numpy as np
from itertools import combinations

def cardValue(num):
    return int(np.ceil(num / 4))

# Suit: Spade > Heart > Diamond > Club
suit_order = {1: 2, 2: 1, 3: 3, 0: 4}

def card_rank(num):
    return int(np.ceil(num / 4))

def card_suit(num):
    return num % 4

def compare_cards(a, b):
    rank_diff = card_rank(a) - card_rank(b)
    if rank_diff != 0:
        return rank_diff
    return suit_order[card_suit(a)] - suit_order[card_suit(b)]

def isPair(hand):
    return hand.size == 2 and card_rank(hand[0]) == card_rank(hand[1])

def isFullHouse(hand):
    if hand.size != 5:
        return False
    hand = hand[np.argsort([card_rank(c) for c in hand])]
    ranks = [card_rank(c) for c in hand]
    return (ranks[0] == ranks[1] == ranks[2] and ranks[3] == ranks[4]) or \
           (ranks[0] == ranks[1] and ranks[2] == ranks[3] == ranks[4])

def isFourOfAKind(hand):
    if hand.size != 5:
        return False
    ranks = [card_rank(c) for c in hand]
    for r in set(ranks):
        if ranks.count(r) == 4:
            return True
    return False

def isStraight(hand):
    if hand.size != 5:
        return False
    ranks = sorted([card_rank(c) for c in hand])
    unique_ranks = sorted(list(set(ranks)))
    if len(unique_ranks) != 5:
        return False
    if unique_ranks == [11, 12, 13, 1, 2]:  # JQKA2
        return True
    for i in range(4):
        if unique_ranks[i] + 1 != unique_ranks[i + 1]:
            return False
    return 3 <= unique_ranks[0] <= 10

def isRealHand(hand):
    if hand.size == 1:
        return True
    elif hand.size == 2:
        return isPair(hand)
    elif hand.size == 5:
        return isStraight(hand) or isFullHouse(hand) or isFourOfAKind(hand)
    return False

def validatePlayedHand(hand, prevHand, control):
    if not isRealHand(hand):
        return False
    if control == 1:
        return True
    if hand.size != prevHand.size:
        return False

    hand = np.sort(hand)
    prevHand = np.sort(prevHand)

    if hand.size == 1:
        return compare_cards(hand[0], prevHand[0]) > 0

    elif hand.size == 2:
        if not isPair(hand):
            return False
        return compare_cards(hand[1], prevHand[1]) > 0

    elif hand.size == 5:
        if isFourOfAKind(hand):
            if not isFourOfAKind(prevHand):
                return True
            return max(hand, key=card_rank) > max(prevHand, key=card_rank)

        if isFullHouse(hand):
            if isFourOfAKind(prevHand):
                return False
            if not isFullHouse(prevHand):
                return True
            return max([card_rank(c) for c in hand]) > max([card_rank(c) for c in prevHand])

        if isStraight(hand):
            if isFourOfAKind(prevHand) or isFullHouse(prevHand):
                return False
            if not isStraight(prevHand):
                return True
            return max([card_rank(c) for c in hand]) > max([card_rank(c) for c in prevHand])

    return False

def convertHand(hand):
    output = np.zeros(len(hand))
    for i, card in enumerate(hand):
        if card[0] == "2":
            base = 13
        elif card[0] == "A":
            base = 12
        elif card[0] == "K":
            base = 11
        elif card[0] == "Q":
            base = 10
        elif card[0] == "J":
            base = 9
        elif card[0] == "1":
            base = 8
            card = card.replace("0", "")
        else:
            base = int(card[0]) - 2

        suit_char = card[-1]
        if suit_char == "D":
            suit = 1
        elif suit_char == "C":
            suit = 2
        elif suit_char == "H":
            suit = 3
        elif suit_char == "S":
            suit = 0

        output[i] = int((base - 1) * 4 + suit)
    return output

class card:
    def __init__(self, number, i):
        self.suit = number % 4
        self.value = np.ceil(number / 4)
        self.indexInHand = i
        self.inPair = 0
        self.inThreeOfAKind = 0
        self.inFourOfAKind = 0
        self.inFlush = 0
        self.inStraight = 0
        self.straightIndex = -1
        self.flushIndex = -1

    def __repr__(self):
        if self.value < 8:
            string1 = str(int(self.value + 2))
        elif self.value == 8:
            string1 = "10"
        elif self.value == 9:
            string1 = "J"
        elif self.value == 10:
            string1 = "Q"
        elif self.value == 11:
            string1 = "K"
        elif self.value == 12:
            string1 = "A"
        elif self.value == 13:
            string1 = "2"

        suit_map = {1: "D", 2: "C", 3: "H", 0: "S"}
        string2 = suit_map[self.suit]
        return f"<card. {string1 + string2} inPair: {self.inPair}, inThree: {self.inThreeOfAKind}, inFlush: {self.inFlush}, inStraight: {self.inStraight}>"

class handsAvailable:
    def __init__(self, currentHand):
        self.cHand = np.sort(currentHand).astype(int)
        self.handLength = currentHand.size
        self.cards = {}
        for i in range(self.cHand.size):
            self.cards[self.cHand[i]] = card(self.cHand[i], i)
        self.flushes = []
        self.pairs = []
        self.threeOfAKinds = []
        self.fourOfAKinds = []
        self.straights = []
        self.nPairs = 0
        self.nThreeOfAKinds = 0
        self.nDistinctPairs = 0
        self.fillPairs()
        self.fillStraights()
        self.fillFullHouses()
        self.fillFourOfAKinds()

    def fillPairs(self):
        seen = set()
        for comb in combinations(self.cHand, 2):
            if isPair(np.array(comb)):
                val = card_rank(comb[0])
                if val not in seen:
                    seen.add(val)
                    self.nDistinctPairs += 1
                self.pairs.append(comb)
                self.nPairs += 1
                for c in comb:
                    self.cards[c].inPair = 1

    def fillStraights(self):
        for comb in combinations(self.cHand, 5):
            if isStraight(np.array(comb)):
                self.straights.append(comb)
                for c in comb:
                    self.cards[c].inStraight = 1

    def fillFullHouses(self):
        for comb in combinations(self.cHand, 5):
            if isFullHouse(np.array(comb)):
                self.threeOfAKinds.append(comb)
                for c in comb:
                    self.cards[c].inThreeOfAKind = 1

    def fillFourOfAKinds(self):
        for comb in combinations(self.cHand, 5):
            if isFourOfAKind(np.array(comb)):
                self.fourOfAKinds.append(comb)
                for c in comb:
                    self.cards[c].inFourOfAKind = 1
