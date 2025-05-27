from channels.generic.websocket import AsyncWebsocketConsumer
import json
from big2Game import big2Game
import numpy as np
import random
from gameLogic import CardPlay, PlayType, PlayerHandCard
from math import comb
import copy

class minimaxBig2Sim():
    playHistory: list[CardPlay]
    agent_index: int
    
    myHand:   PlayerHandCard
    oppoHand: PlayerHandCard
    oppoCardCount: int[3]
    # reset() not used
    # assignRewards() not used
    # modifed
    def __init__(self, agent_index:int):
        self.setAgentIndex(agent_index)
    # add
    # renew agent_index
    def setAgentIndex(self, agent_index:int):
        self.agent_index = agent_index
    # add
    # load current big2Game table into simulation
    def load(self, game:big2Game):
        self.myHand = game.PlayersHand[self.agent_index]
        self.playersGo = game.playersGo
        self.playHistory = game.playHistory
        
        self.oppoHand = PlayerHandCard([cards for player in self.PlayersHand if player is not self.PlayersHand[self.agent_index] for cards in player.get_cards_index()])
        self.oppoCardCount = []
        cards = []
        oppo_index = self.agent_index
        for _ in range(3):
            oppo_index = (oppo_index + 1) % 4
            cards.append(self.PlayersHand[oppo_index])
            self.oppoCardCount.append(self.PlayersHand[oppo_index].__len__())
    # add
    # -1 if it's my turn, [0,2] if it's others' turn
    def index(self):
        index = (self.playersGo - self.agent_index) % 4 - 1
        return index
    # add
    def my_win(self):
        return self.myHand.__len__() == 0
    # modified
    def step(self, play: CardPlay):
        if self.playersGo == self.agent_index:
            if not self.myHand.test_play(play, self.getPlayOnTop(), self.playerHasControl()):
                raise ValueError("Invalid play")
            self.myHand.remove_played_cards(play)
            self.playHistory.append(play)
            if self.isGameOver():
                self.assignRewards()
                return
            self.playersGo = (self.playersGo + 1) % 4
        else:
            if not self.oppoHand.test_play(play, self.getPlayOnTop(), self.playerHasControl()):
                raise ValueError("Invalid play")
            self.oppoHand.remove_played_cards(play)
            self.oppoCardCount[self.index()] -= play.__len__()
            self.playHistory.append(play)
            if self.isGameOver():
                self.assignRewards()
                return
            self.playersGo = (self.playersGo + 1) % 4
    # same as big2Game
    def getHistory(self, length: int | None = None, no_pass: bool = False):
        """
        Get the last n plays in the play history.
        If length is None, return the entire play history.
        If history is shorter than length, pad with empty plays.
        """
        if no_pass:
            history = [i for i in self.playHistory if i.get_type() != PlayType.PASS]
        else:
            history = self.playHistory

        if length is None:
            return history
        if length < 1:
            raise ValueError("length must be greater than 0")
        if len(history) < length:
            res = []
            for _ in range(length - len(history)):
                res.append(CardPlay([]))
            res.extend(history)
            return res
        else:
            return history[-length:]
    # modified
    def isGameOver(self):
        return self.myHand.__len__() == 0 or any(self.oppoCardCount[i] == 0 for i in range(3))
    # same as big2Game
    def playerHasControl(self):
        return all(i.get_type() == PlayType.PASS for i in self.getHistory(length=3))
    # same as big2Game
    def getPlayOnTop(self):
        for prev_play in self.playHistory[::-1]:
            if prev_play.get_type() != PlayType.PASS:
                return prev_play
    # modified
    def getCurrentState(self):
        return (
            self.playersGo,
            self.getHistory(),
            self.myHand           if self.playersGo == self.agent_index else self.oppoHand,
            self.myHand.__len__() if self.playersGo == self.agent_index else self.oppoCardCount[self.index()]
        )
    # modified (don't know if it's useful or not)
    def getInfoForDrawing(self):
        info = {
            "type": "updateGame",
            "myHand": self.myHand.get_cards_index(),
            "oppoHand": self.oppoHand.get_cards_index(),
            "oppoCardCount": self.oppoCardCount,
            "playersGo": self.playersGo + 1,  # 1~4
            "control": self.playerHasControl(),
            "previousHands": [
                i.get_cards_index() for i in self.getHistory(length=3, no_pass=True)
            ],
            "gameOver": self.isGameOver()
        }
        return info
# simulate a play by deep copy a simulation table and step once
def sim_play(table: minimaxBig2Sim, play: CardPlay):
    next = copy.deepcopy(table)
    next.step(play)
    return next

def score_play(table: minimaxBig2Sim, play: CardPlay):
    next_table = sim_play(table, play)
    return get_heuristic(next_table)

play_weight = {
    # [card count, ]
    PlayType.UNKNOWN :         0,  # 無效牌型
    PlayType.SINGLE :          1,  # 單牌
    PlayType.PAIR :            2,  # 對子，兩張同點數
    PlayType.THREE_OF_A_KIND : 3,  # 三條，三張同點數
    PlayType.STRAIGHT :        5,  # 順子，五張連續點數
    PlayType.FLUSH :           5,  # 同花，五張同花色
    PlayType.FULL_HOUSE :      5,  # 葫蘆，三條加一對
    PlayType.FOUR_OF_A_KIND :  5,  # 四條，四張同點數和一張其他牌
    PlayType.STRAIGHT_FLUSH :  5,  # 同花順，五張連續點數且同花色
    PlayType.PASS :            0   # 不出牌
}

def get_heuristic(table: minimaxBig2Sim):
    if table.my_win(): return 1e7        # win
    elif table.isGameOver(): return -1e7 # lose
    """
    depth reached(game isn't over)
    factors: control and card plays
    Am I in control when depth reached? or who's next?
    threat and risk opponents pose (larger play probability * disturbance for me)
    """
    hand_card_count = sum(table.myHand.__len__(), table.oppoCardCount)
    hand_cards = 
    if table.playersGo == table.agent_index:
        h_value = table.playerHasControl() * 1e5
    else:
    return h_value

class MinimaxAgent:
    sim: minimaxBig2Sim
    def __init__(self, hand_card, agent_index, depth = 2):
        self.hand_card = hand_card
        self.agent_index = agent_index
        self.sim = minimaxBig2Sim(agent_index)
        self.depth = depth
    ##### unfinished #####
    # Would be nicer if we have this, I suppose
    # Make a custom recipe to manipulate the depth to avoid over-calculation
    def dynamic_depth(self):
        return
    # Called minimax agent to step
    def step(self, game: big2Game):
        player_go, first_player, history, hand, available_actions = game.getCurrentState()
        if len(available_actions) == 0:
            raise ValueError("No available actions")
        self.sim.load(game)
        _, best_action = self.minimax(self.sim, self.depth)
        action = random.choice(best_action)
        return action
    # recursive minimax with alpha-beta
    def minimax(self, game_sim: minimaxBig2Sim, depth, alpha, beta):
        """
        Goal:   Implement recursive Minimax search for Big2.
        Return: (tableValue, {setOfCandidatePlays})
            - tableValue is the evaluated utility of the table state
            - {setOfCandidatePlays} is a set of actions that achieve this tableValue
        """
        if depth == 0 or game_sim.isGameOver():
            return get_heuristic(game_sim), set()
    
        val = []
        if game_sim.playersGo == game_sim.agent_index:
            result = -2e10
            available_plays = game_sim.myHand.get_available_plays()
            for play in available_plays:
                value = self.minimax(sim_play(game_sim, play), depth - 1, alpha, beta)[0]
                val.append(value)
                result = max(result, value)
                if result >= beta: break
                alpha = max(alpha, result)
        else:
            result = 2e10
            available_plays = game_sim.myHand.get_available_plays()
            for play in available_plays:
                if (play.__len__() > game_sim.oppoCardCount[game_sim.index()]):
                    val.append(2e10)
                    continue
                value = self.minimax(sim_play(game_sim, play), depth - 1, alpha, beta)[0]
                val.append(value)
                result = min(result, value)
                if alpha >= result: break
                beta = min(beta, result)
            
        # Placeholder return to keep function structure intact
        return result, {play for play, v in zip(available_plays, val) if v == result}