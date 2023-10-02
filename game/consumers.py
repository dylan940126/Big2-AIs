from channels.generic.websocket import WebsocketConsumer
import json
from . import big2Game
from . import enumerateOptions
from . import PPONetwork
import tensorflow.compat.v1 as tf
import joblib
import os
from django.conf import settings

tf.disable_v2_behavior()
sess = tf.Session()
trainedAINetwork = PPONetwork.PPONetwork(sess, 412, 1695, "trainedNetwork")
tf.global_variables_initializer().run(session=sess)
params = joblib.load(os.path.join(settings.BASE_DIR, "modelParameters136500"))
trainedAINetwork.loadParams(params)

print("LOADED NETWORK!!!")

class GameConsumer(WebsocketConsumer):
    def connect(self):
        self.game = big2Game.big2Game()
        #send initial state - write a function big2Game.infoForDrawing
        self.accept()
        self.sendCurrentGameState()
        
    def disconnect(self, close_code):
        pass
        
    def sendCurrentGameState(self):
        self.send(text_data=json.dumps(self.game.getInfoForDrawing()))
        
    def receive(self, text_data):
        #when we receive something from client side.
        data = json.loads(text_data)
        #import pdb; pdb.set_trace()
        if data['type'] == "AIGo":
            pGo, state, availAcs = self.game.getCurrentState()
            action, v, nlp = trainedAINetwork.step(state, availAcs)
            reward, done, info = self.game.step(action[0])
            self.sendCurrentGameState()
        elif data['type'] == "reset":
            self.game.reset()
            self.sendCurrentGameState()
        elif data['type'] == "pass":
            if self.game.playersGo != 1:
                self.send(text_data=json.dumps({
                    'type' : "error",
                    'error' : "Wasn't your go ya cheat!"
                }))
            else:
                reward, done, info = self.game.step(enumerateOptions.passInd)
                self.sendCurrentGameState()
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
                self.send(text_data=json.dumps({
                    'type' : "error",
                    'error' : "Server says no. Stop trying to cheat!"
                }))
            else:
                reward, done, info = self.game.step(index)
                self.sendCurrentGameState()
            