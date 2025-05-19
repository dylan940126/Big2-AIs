from channels.generic.websocket import AsyncWebsocketConsumer
import json
from big2Game import big2Game
import random
import numpy as np
from cnnAgent import CNNBot, hand_to_matrix, play_to_matrix
from cardEstimator import estimate_opponent_cards  # <- use dummy for now


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

cnn_agent = CNNBot(model_path="cnn_model.pt", device="cpu")  # Adjust path/device


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
        """
        if data["type"] == "AIGo":
            pGo, history, availAcs = self.game.getCurrentState()
            print(
                f"Player: {pGo}, Hand: {self.game.PlayersHand[pGo]}, Available Actions: {availAcs}"
            )
            action = random_agent.step(history, availAcs)
            self.game.step(action)
            await self.sendCurrentGameState()
        """
        if data["type"] == "AIGo":
            pGo, history, availAcs = self.game.getCurrentState()

            if len(availAcs) == 0:
                self.game.step(big2Game.CardPlay([]))
                await self.sendCurrentGameState()
                return

            hand_matrix = hand_to_matrix(self.game.PlayersHand[pGo])

            played_cards = [p for p in self.game.playHistory if p.get_type() != big2Game.PlayType.PASS]
            remaining_counts = [len(self.game.PlayersHand[i]) for i in range(4) if i != pGo]

            predicted_matrix_3 = estimate_opponent_cards(
                current_hand=hand_matrix,
                played_cards=played_cards,
                remaining_counts=remaining_counts,
                history=self.game.playHistory
            )

            predicted_matrix = np.mean(predicted_matrix_3, axis=0)  # collapse 3x4x13 to 4x13

            try:
                action = cnn_agent.step(predicted_matrix, hand_matrix, availAcs)
                self.game.step(action)
            except ValueError as e:
                await self.send(text_data=json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                return

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
        elif data["type"] == "autoPlay":
            self.games_played = 0
            self.total_wins = [0, 0, 0, 0]
            self.total_rewards = [0, 0, 0, 0]
            await self.runAutoPlay(num_games=1000)

    async def runAutoPlay(self, num_games):
        while self.games_played < num_games:
            if self.game.isGameOver():
                self.game.assignRewards()
                winner = int(np.argmax(self.game.rewards))
                self.total_wins[winner] += 1
                for i in range(4):
                    self.total_rewards[i] += self.game.rewards[i]

                self.games_played += 1
                self.game.reset()
                continue

            pGo, history, availAcs = self.game.getCurrentState()

            if len(availAcs) == 0:
                self.game.step(big2Game.CardPlay([]))
                continue

            hand_matrix = hand_to_matrix(self.game.PlayersHand[pGo])
            played_cards = [p for p in self.game.playHistory if p.get_type() != big2Game.PlayType.PASS]
            remaining_counts = [len(self.game.PlayersHand[i]) for i in range(4) if i != pGo]

            predicted_matrix_3 = estimate_opponent_cards(
                current_hand=hand_matrix,
                played_cards=played_cards,
                remaining_counts=remaining_counts,
                history=self.game.playHistory
            )

            predicted_matrix = np.mean(predicted_matrix_3, axis=0)

            try:
                action = cnn_agent.step(predicted_matrix, hand_matrix, availAcs)
                self.game.step(action)
            except ValueError:
                self.game.step(big2Game.CardPlay([]))

        # Done with all games
        await self.send(text_data=json.dumps({
            "type": "autoDone",
            "message": f"{num_games} games completed.",
            "totalWins": self.total_wins,
            "totalRewards": self.total_rewards,
            "winRates": [round(w / num_games, 3) for w in self.total_wins]
        }))
