function isRealHand(hand) {
  if (hand.length > 5 || hand.length == 0) return 0;
  if (hand.length == 1) return 1;
  if (hand.length == 2 && isPair(hand)) return 1;
  if (hand.length == 3 && isThreeOfAKind(hand)) return 1;
  if (hand.length == 4 && (isFourOfAKind(hand) || isTwoPair(hand))) return 1;
  if (
    hand.length == 5 &&
    (isStraight(hand) ||
      isFlush(hand) ||
      isFullHouse(hand) ||
      isStraightFlush(hand))
  )
    return 1;
  return 0;
}

function isPair(hand) {
  //takes an array and returns 0 if not a pair, 1 if a pair.
  if (hand.length != 2) return 0;
  if (Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4)) {
    return 1;
  } else {
    return 0;
  }
}

function isThreeOfAKind(hand) {
  //returns 0 if not 3 of a kind or 1 if it is.
  if (hand.length != 3) return 0;
  if (
    Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[0] / 4) == Math.ceil(hand[2] / 4)
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isFourOfAKind(hand) {
  //returns 0 if not 3 of a kind or 1 if it is.
  if (hand.length != 4) return 0;
  if (
    Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[0] / 4) == Math.ceil(hand[2] / 4) &&
    Math.ceil(hand[0] / 4) == Math.ceil(hand[3] / 4)
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isTwoPair(hand) {
  if (hand.length != 4) return 0;
  hand.sort(sortNumber);
  if (
    Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[2] / 4) == Math.ceil(hand[3] / 4) &&
    !isFourOfAKind(hand)
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isStraightFlush(hand) {
  if (hand.length != 5) return 0;
  hand.sort(sortNumber);
  if (
    hand[0] + 4 == hand[1] &&
    hand[1] + 4 == hand[2] &&
    hand[2] + 4 == hand[3] &&
    hand[3] + 4 == hand[4]
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isStraight(hand) {
  if (hand.length != 5) return 0;
  hand.sort(sortNumber);
  if (
    Math.ceil(hand[0] / 4) + 1 == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[1] / 4) + 1 == Math.ceil(hand[2] / 4) &&
    Math.ceil(hand[2] / 4) + 1 == Math.ceil(hand[3] / 4) &&
    Math.ceil(hand[3] / 4) + 1 == Math.ceil(hand[4] / 4) &&
    !isStraightFlush(hand)
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isFlush(hand) {
  if (hand.length != 5) return 0;
  if (
    hand[0] % 4 == hand[1] % 4 &&
    hand[0] % 4 == hand[2] % 4 &&
    hand[0] % 4 == hand[3] % 4 &&
    hand[0] % 4 == hand[4] % 4 &&
    !isStraightFlush(hand)
  ) {
    return 1;
  } else {
    return 0;
  }
}

function isFullHouse(hand) {
  //returns 0 if not full house. 1 if fullhouse with smaller value being the 3 cards, 2 if fullhouse with larger value being the 3 cards.
  if (hand.length != 5) return 0;
  hand.sort(sortNumber);
  if (
    Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[0] / 4) == Math.ceil(hand[2] / 4) &&
    Math.ceil(hand[3] / 4) == Math.ceil(hand[4] / 4)
  ) {
    return 1;
  } else if (
    Math.ceil(hand[0] / 4) == Math.ceil(hand[1] / 4) &&
    Math.ceil(hand[2] / 4) == Math.ceil(hand[3] / 4) &&
    Math.ceil(hand[2] / 4) == Math.ceil(hand[4] / 4)
  ) {
    return 2;
  }
}

function compareHands(hand1, hand2) {
  //returns 1 if hand1 beats hand2, 0 if hand2 beats hand1.
  //assume hands are valid.
  hand1.sort(sortNumber);
  hand2.sort(sortNumber);
  if (hand1.length == 1) {
    if (hand1[0] > hand2[0]) {
      return 1;
    } else {
      return 0;
    }
  } else if (hand1.length == 2) {
    if (hand1[1] > hand2[1]) {
      return 1;
    } else {
      return 0;
    }
  } else if (hand1.length == 3) {
    if (hand1[1] > hand2[1]) {
      return 1;
    } else {
      return 0;
    }
  } else if (hand1.length == 4) {
    if (isFourOfAKind(hand1) && !isFourOfAKind(hand2)) {
      return 1;
    } else if (!isFourOfAKind(hand1) && isFourOfAKind(hand2)) {
      return 0;
    } else if (isFourOfAKind(hand1) && isFourOfAKind(hand2)) {
      if (hand1[1] > hand2[1]) {
        return 1;
      } else {
        return 0;
      }
    } else {
      //both two pair. Hand with highest pair wins.
      if (hand1[3] > hand2[3]) {
        return 1;
      } else {
        return 0;
      }
    }
  } else {
    //is a 5 card hand.
    if (
      isStraight(hand1) &&
      (isFlush(hand2) || isFullHouse(hand2) || isStraightFlush(hand2))
    ) {
      return 0;
    } else if (
      (isFlush(hand1) || isFullHouse(hand1) || isStraightFlush(hand1)) &&
      isStraight(hand2)
    ) {
      return 1;
    } else if (
      isFlush(hand1) &&
      (isFullHouse(hand2) || isStraightFlush(hand2))
    ) {
      return 0;
    } else if (
      (isFullHouse(hand1) || isStraightFlush(hand1)) &&
      (isFlush(hand2) || isStraight(hand2))
    ) {
      return 1;
    } else if (isFullHouse(hand1) && isStraightFlush(hand2)) {
      return 0;
    } else if (isStraightFlush(hand1) && isFullHouse(hand2)) {
      return 1;
    } else if (isStraight(hand1) && isStraight(hand2)) {
      if (hand1[4] > hand2[4]) {
        return 1;
      } else {
        return 0;
      }
    } else if (isFlush(hand1) && isFlush(hand2)) {
      if (hand1[4] > hand2[4]) {
        return 1;
      } else {
        return 0;
      }
    } else if (isFullHouse(hand1) && isFullHouse(hand2)) {
      var max1, max2;
      if (isFullHouse(hand1) == 1) {
        max1 = hand1[0];
      } else {
        max1 = hand1[4];
      }
      if (isFullHouse(hand2) == 1) {
        max2 = hand2[0];
      } else {
        max2 = hand2[4];
      }
      if (max1 > max2) {
        return 1;
      } else {
        return 0;
      }
    }
  }
}
