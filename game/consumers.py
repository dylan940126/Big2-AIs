# filepath: /home/dylan/data/文件/NYCU/大二下/515515人工智慧概論/Final Project/game/consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from game.big2Game import big2Game
from game.gameLogic import CardPlay
from agents import Agent, HumanAgent, CNNAgent
from typing import List

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.game = big2Game()
        self.url_route = self.scope.get("url_route", {})
        self.url_args = self.url_route.get("kwargs", {})

        await self.accept()
        await self.sendCurrentGameState()

        human_agent = HumanAgent()
        cnn_agent_1 = CNNAgent(model="cnn_agent_best.pt", train=False)
        cnn_agent_2 = CNNAgent(model=cnn_agent_1.model, train=False)
        cnn_agent_3 = CNNAgent(model=cnn_agent_1.model, train=False)

        self.agents: List[Agent] = [human_agent, cnn_agent_1, cnn_agent_2, cnn_agent_3]

    async def disconnect(self, close_code):
        pass

    async def sendCurrentGameState(self):
        await self.send(text_data=json.dumps(self.game.getInfoForDrawing()))

    async def receive(self, text_data):
        # when we receive something from client side.
        data = json.loads(text_data)
        # import pdb; pdb.set_trace()
        if data["type"] == "AIGo" or data["type"] == "submitPlayerHand":
            if data["type"] == "submitPlayerHand":
                self.agents[0].update_human_action(CardPlay(data["hand"]))
            
            playersGo, firstPlayer, history, hand, availAcs = (
                self.game.getCurrentState()
            )

            agent = self.agents[playersGo]
            try:
                action = agent.step(firstPlayer, history, hand, availAcs)
                self.game.step(action)
            except ValueError as e:
                await self.send(
                    text_data=json.dumps({"type": "error", "error": str(e)})
                )
                return

            await self.sendCurrentGameState()

        elif data["type"] == "reset":
            self.game.reset()
            for agent in self.agents:
                agent.reset()
            await self.sendCurrentGameState()

        # elif data["type"] == "autoPlay":
        #     self.games_played = 0
        #     self.total_wins = [0, 0, 0, 0]
        #     self.total_rewards = [0, 0, 0, 0]
        #     await self.runAutoPlay(num_games=1000)

    # async def runAutoPlay(self, num_games):
    #     while self.games_played < num_games:
    #         if self.game.isGameOver():
    #             self.game.assignRewards()
    #             winner = int(np.argmax(self.game.rewards))
    #             self.total_wins[winner] += 1
    #             for i in range(4):
    #                 self.total_rewards[i] += self.game.rewards[i]

    #             self.games_played += 1
    #             self.game.reset()
    #             continue

    #         pGo, firstPlayer, history, hand, availAcs = self.game.getCurrentState()

    #         if len(availAcs) == 0:
    #             self.game.step(big2Game.CardPlay([]))
    #             continue

    #         # 在這裡也需要根據玩家選擇不同的 agent
    #         try:
    #             if pGo == 1:
    #                 agent = cnn_agent_1
    #             elif pGo == 2:
    #                 agent = cnn_agent_2
    #             elif pGo == 3:
    #                 agent = cnn_agent_3
    #             else:
    #                 raise ValueError(f"未知的玩家編號: {pGo}")

    #             action = agent.step(
    #                 firstPlayer, history, hand, availAcs
    #             )
    #             self.game.step(action)
    #         except ValueError:
    #             self.game.step(big2Game.CardPlay([]))

    #     # Done with all games
    #     await self.send(text_data=json.dumps({
    #         "type": "autoDone",
    #         "message": f"{num_games} games completed.",
    #         "totalWins": self.total_wins,
    #         "totalRewards": self.total_rewards,
    #         "winRates": [round(w / num_games, 3) for w in self.total_wins]
    #     }))
