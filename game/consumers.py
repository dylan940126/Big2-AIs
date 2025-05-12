from channels.generic.websocket import AsyncWebsocketConsumer
import json
from . import big2Game
import random


class RandomAgent:
    def step(self, history, availAcs: list[big2Game.CardPlay]):
        """
        Randomly select an action from the available actions.
        """
        if len(availAcs) == 0:
            raise ValueError("No available actions")
        action = random.choice(availAcs)
        return action


random_agent = RandomAgent()


class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.game = big2Game.big2Game()
        self.url_route = self.scope.get("url_route", {})
        self.url_args = self.url_route.get("kwargs", {})

        await self.accept()
        await self.sendCurrentGameState()

    async def disconnect(self, close_code):
        pass

    async def sendCurrentGameState(self):
        await self.send(text_data=json.dumps(self.game.getInfoForDrawing()))

    async def receive(self, text_data):
        # when we receive something from client side.
        data = json.loads(text_data)
        # import pdb; pdb.set_trace()
        if data["type"] == "AIGo":
            pGo, history, availAcs = self.game.getCurrentState()
            print(
                f"Player: {pGo}, Hand: {self.game.PlayersHand[pGo]}, Available Actions: {availAcs}"
            )
            action = random_agent.step(history, availAcs)
            self.game.step(action)
            await self.sendCurrentGameState()
        elif data["type"] == "reset":
            self.game.reset()
            await self.sendCurrentGameState()
        elif data["type"] == "pass":
            if self.game.playersGo != 0:
                await self.send(
                    text_data=json.dumps(
                        {"type": "error", "error": "Wasn't your go ya cheat!"}
                    )
                )
            else:
                self.game.step(big2Game.CardPlay([]))
                await self.sendCurrentGameState()
        elif data["type"] == "submitPlayerHand":
            # indsPlayed = data['hand']
            # if len(indsPlayed)==1:
            #     opt = indsPlayed[0]
            #     nC = 1
            # elif len(indsPlayed)==2:
            #     opt = enumerateOptions.twoCardIndices[indsPlayed[0]][indsPlayed[1]]
            #     nC = 2
            # elif len(indsPlayed)==3:
            #     opt = enumerateOptions.threeCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]]
            #     nC = 3
            # elif len(indsPlayed)==4:
            #     opt = enumerateOptions.fourCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]][indsPlayed[3]]
            #     nC = 4
            # elif len(indsPlayed)==5:
            #     opt = enumerateOptions.fiveCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]][indsPlayed[3]][indsPlayed[4]]
            #     nC = 5
            # index = enumerateOptions.getIndex(opt, nC)
            # availAcs = self.game.returnAvailableActions()
            # if availAcs[index] == 0:
            #     await self.send(text_data=json.dumps({
            #         'type' : "error",
            #         'error' : "Server says no. Stop trying to cheat!"
            #     }))
            # else:
            #     reward, done, info = self.game.step(index)
            #     await self.sendCurrentGameState()
            play = data["hand"]
            if self.game.playersGo != 0:
                await self.send(
                    text_data=json.dumps(
                        {"type": "error", "error": "Wasn't your go ya cheat!"}
                    )
                )
            else:
                try:
                    self.game.step(big2Game.CardPlay(play))
                    await self.sendCurrentGameState()
                except ValueError as e:
                    await self.send(
                        text_data=json.dumps({"type": "error", "error": str(e)})
                    )
