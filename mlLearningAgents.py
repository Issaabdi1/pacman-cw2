# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
from collections import defaultdict
from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    def __init__(self, state: GameState):
        self.state = state
        self.pacmanPosition = state.getPacmanPosition()
        self.ghostPositions = tuple(state.getGhostPositions())  # Convert to tuple
        self.food = tuple(map(tuple, state.getFood()))  # Convert Grid to a hashable structure
        self.score = state.getScore()
    
    def __hash__(self):
        return hash((self.pacmanPosition, self.ghostPositions, self.food))

class QLearnAgent(Agent):
    def __init__(self, alpha=0.2, epsilon=0.2, gamma=0.8, maxAttempts=30, numTraining=2000):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon  # Slower decay
        self.gamma = gamma
        self.maxAttempts = maxAttempts
        self.numTraining = numTraining
        self.episodesSoFar = 0
        self.qValues = defaultdict(lambda: defaultdict(float))  # Ensure persistence
        self.visitationCounts = defaultdict(lambda: defaultdict(int))
    
    def computeReward(self, startState: GameState, endState: GameState) -> float:
        if endState.isWin():
            return 100  # Large reward for winning
        elif endState.isLose():
            return -200  # Less severe penalty to allow better exploration
        
        reward = -1  # Small step penalty to encourage efficiency

        # Increase reward for eating food
        startFood = startState.getNumFood()
        endFood = endState.getNumFood()
        if endFood < startFood:
            reward += 100  # Stronger incentive for eating food

        # Penalize proximity to ghosts more strongly
        pacmanPos = endState.getPacmanPosition()
        for ghost in endState.getGhostPositions():
            distance = util.manhattanDistance(pacmanPos, ghost)
            if distance == 0:
                reward -= 500  # Massive penalty for losing immediately
            elif distance < 2:
                reward -= 50  # Large penalty for being too close
            elif distance < 3:
                reward -= 10  # Mild penalty

        return reward
    
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        return self.qValues[state][action]
    
    def maxQValue(self, state: GameStateFeatures) -> float:
        return max(self.qValues[state].values(), default=0)
    
    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        bestNextQ = self.maxQValue(nextState)
        oldQValue = self.qValues[state][action]
        self.qValues[state][action] = (1 - self.alpha) * oldQValue + self.alpha * (reward + self.gamma * bestNextQ)
        
        # Debugging: Print Q-value updates
        print(f"Updated Q-value: Q({state}, {action}) = {self.qValues[state][action]:.2f}")
    
    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.visitationCounts[state][action] += 1
    
    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        return self.visitationCounts[state][action]
    
    def explorationFn(self, utility: float, counts: int) -> float:
        return utility + (2.0 / (1.0 + counts))  # More aggressive exploration boost
    
    def getAction(self, state: GameState) -> Directions:
    # Get legal actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Convert state to features
        stateFeatures = GameStateFeatures(state)

        # Exploration vs. Exploitation (Epsilon-Greedy)
        if util.flipCoin(self.epsilon):  # Exploration
            action = random.choice(legal)
        else:  # Exploitation
            action = max(legal, key=lambda a: self.getQValue(stateFeatures, a))

        # Get next state after taking action
        nextState = state.generatePacmanSuccessor(action)
        nextStateFeatures = GameStateFeatures(nextState)

        # Compute reward
        reward = self.computeReward(state, nextState)

        # Perform Q-learning update
        self.learn(stateFeatures, action, reward, nextStateFeatures)

        # Reduce exploration (decay epsilon)
        self.epsilon = max(0.1, self.epsilon * 0.99)  # More gradual epsilon decay

        return action
    
    def getEpisodesSoFar(self):
        return self.episodesSoFar
    
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1
    
    def final(self, state: GameState):
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.numTraining:
            print('Training Done (turning off epsilon and alpha)')
            self.alpha = 0.01  # Keep small learning rate to allow slight adjustments
            self.epsilon = 0.05  # Keep some exploration for unexpected situations

