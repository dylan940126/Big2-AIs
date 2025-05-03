from channels.generic.websocket import AsyncWebsocketConsumer
import json
from . import big2Game
from . import enumerateOptions
import random

class RandomAgent:
    def step(self, obs, availAcs):
        # availAcs: list of available actions, 1 for available, 0 or -inf for unavailable
        # 這裡假設 availAcs 是一個 list 或 numpy array
        available_indices = [i for i, v in enumerate(availAcs[0]) if v == 0]
        if not available_indices:
            return [0], None, None  # fallback
        action = random.choice(available_indices)
        return [action], None, None

random_agent = RandomAgent()

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.game = big2Game.big2Game()
        #send initial state - write a function big2Game.infoForDrawing
        await self.accept()
        await self.sendCurrentGameState()
        
    async def disconnect(self, close_code):
        pass
        
    async def sendCurrentGameState(self):
        await self.send(text_data=json.dumps(self.game.getInfoForDrawing()))
        
    async def receive(self, text_data):
        #when we receive something from client side.
        data = json.loads(text_data)
        #import pdb; pdb.set_trace()
        if data['type'] == "AIGo":
            pGo, state, availAcs = self.game.getCurrentState()
            action, v, nlp = random_agent.step(state, availAcs)
            reward, done, info = self.game.step(action[0])
            await self.sendCurrentGameState()
        elif data['type'] == "reset":
            self.game.reset()
            await self.sendCurrentGameState()
        elif data['type'] == "pass":
            if self.game.playersGo != 1:
                await self.send(text_data=json.dumps({
                    'type' : "error",
                    'error' : "Wasn't your go ya cheat!"
                }))
            else:
                reward, done, info = self.game.step(enumerateOptions.passInd)
                await self.sendCurrentGameState()
        elif data['type'] == "submitPlayerHand":
            indsPlayed = data['hand']
            if len(indsPlayed)==1:
                opt = indsPlayed[0]
                nC = 1
            elif len(indsPlayed)==2:
                opt = enumerateOptions.twoCardIndices[indsPlayed[0]][indsPlayed[1]]
                nC = 2
            elif len(indsPlayed)==3:
                opt = enumerateOptions.threeCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]]
                nC = 3
            elif len(indsPlayed)==4:
                opt = enumerateOptions.fourCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]][indsPlayed[3]]
                nC = 4
            elif len(indsPlayed)==5:
                opt = enumerateOptions.fiveCardIndices[indsPlayed[0]][indsPlayed[1]][indsPlayed[2]][indsPlayed[3]][indsPlayed[4]]
                nC = 5
            index = enumerateOptions.getIndex(opt, nC)
            availAcs = self.game.returnAvailableActions()
            if availAcs[index] == 0:
                await self.send(text_data=json.dumps({
                    'type' : "error",
                    'error' : "Server says no. Stop trying to cheat!"
                }))
            else:
                reward, done, info = self.game.step(index)
                await self.sendCurrentGameState()
