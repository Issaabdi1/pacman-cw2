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
from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
   
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """
    def __init__(self, state: GameState):
        # Extract Pacman's position ( a tuple like (x, y))
        self.pacmanPos = state.getPacmanPosition() # calls state.getPacmanPosition() to store Pacman’s current coordinates.

        # Extract ghost positions (sorted for consistency)
        self.ghostPositions = tuple(sorted(state.getGhostPositions())) # calls state.getGhostPositions() and sorts them

        # Convert the food grid into an immutable(and hashable) tuple of tuples.
        # state.getFood() returns a 2D grid
        foodGrid = state.getFood()
        self.food = tuple(tuple(row) for row in foodGrid)

    #Combines the features (Pacman’s position, ghost positions, food grid)
    def __hash__(self):
        return hash((self.pacmanPos, self.ghostPositions, self.food))

    #use objects as keys in dictionaries
    def __eq__(self, other):
        return (self.pacmanPos, self.ghostPositions, self.food) == (other.pacmanPos, other.ghostPositions, other.food)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.1,
                 epsilon: float = 0.05,
                 gamma: float = 0.7,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Dictionaries to store Q-values and counts.
        # Keys are tuples of the form (stateFeatures, action)
        self.qValues = {}
        self.counts = {} #dictionary to store how many times each state–action pair has been visited

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining
    
    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    
    #check this section again
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        
        """
        Computes the reward for transitioning from startState to endState.
        simply the difference in game score.
        """
        return endState.getScore() - startState.getScore()
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        
        """
        Return Q(state, action) pair from dictionary; if not seen before, default to 0.0.
        """
        return self.qValues.get((state, action), 0.0)
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        """
        Compute the maximum Q-value over all possible actions
        to consider the four main directions
        """
        #method computes the maximum Q–value among all potential actions from the given state
        #It iterates over the four primary directions, retrieves each Q–value, and returns the maximum. 
        #This is used in the update rule to approximate the best future value.
        possibleActions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        qvals = [self.getQValue(state, action) for action in possibleActions]
        if not qvals:
            return 0.0
        return max(qvals)
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        """
        Performs a Q-learning update:
          Q(s,a) = (1 - alpha)*Q(s,a) + alpha*(reward + gamma * max_a' Q(s',a'))
        and updates the visitation count.
        """
        oldQ = self.getQValue(state, action)
        maxNextQ = self.maxQValue(nextState)
        newQ = (1 - self.alpha) * oldQ + self.alpha * (reward + self.gamma * maxNextQ)
        self.qValues[(state, action)] = newQ
        # Update the count for this state-action pair.
        self.updateCount(state, action)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        """
        Increment the visitation count for the (state, action) pair.
        """
        self.counts[(state, action)] = self.counts.get((state, action), 0) + 1 #Increments the stored count for a specific (state, action) pair
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        """
        Returns the count for how many times (state, action) has been visited.
        """
        return self.counts.get((state, action), 0)#Retrieves the current count; if the pair isn’t in the dictionary, it returns 0
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        
        """
        Exploration function that returns a high bonus value for actions
        that have been tried fewer than maxAttempts times.
        """
        if counts < self.maxAttempts:
            # Provide a large bonus to encourage exploration.
            return 1e6#greed pick
        else:
            return utility
        #least pick
        #bonus = 1.0 / (counts + 1)  # The bonus decreases as counts increase.
        #return utility + bonus

    ## WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        """
        Choose an action using epsilon-greedy exploration.
        With probability epsilon, take a random legal move; otherwise,
        pick the move with the highest exploration-adjusted Q-value.
        """
        # Get the legal actions (excluding STOP)
        legal = state.getLegalPacmanActions() #Retrieves a list of legal moves from the current game state, removing the STOP action
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        #Prints key state information (e.g., positions, food, score) to help during development
        # Log information (helpful during development)
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        # Wrap the state in a features object
        stateFeatures = GameStateFeatures(state)#Creates a GameStateFeatures object from the current state

        # Epsilon-greedy decision: sometimes choose a random action.
        if util.flipCoin(self.epsilon):
            chosen = random.choice(legal)#With probability epsilon, a random legal action is chosen
        else:
            bestAction = None
            bestValue = float('-inf')
            for action in legal:
                # Compute exploration function value based on Q-value and count.
                q = self.getQValue(stateFeatures, action)
                cnt = self.getCount(stateFeatures, action)
                value = self.explorationFn(q, cnt)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            chosen = bestAction if bestAction is not None else random.choice(legal)
        return chosen #chosen action is returned, which tells Pacman which move to execute

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
