# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        foodDistances=[]
        count = 0

        #Calculamos las distancias Manhattan de las comidas
        for item in foodList:
            foodDistances.append(manhattanDistance(newPos, item))

        #La comida que está más cerca tendrá más preferencia que las que están más alejadas
        for i in foodDistances:
            if i <= 4:
                count += 1
            elif i > 4 and i <= 10:
                count += 0.5
            else:
                count += 0.25

        ghostDistances = []
        #Calculamos las distancias Manhattan de los fantasmas
        for ghost in successorGameState.getGhostPositions():
            ghostDistances.append(manhattanDistance(ghost, newPos))

        #El fantasma que está más cerca tendrá menos preferencia que los estados en los que los
        #fantasmas están más alejados
        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos:
                count = 5 - count
            elif manhattanDistance(ghost, newPos) <= 4:
                count = 2.5 - count

        return successorGameState.getScore() + count

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, depth,agent):

            # Comprobar si es un estado terminal o se ha llegado a la profundidad maxima
            if (not gameState.getLegalActions(agent) or (depth == self.depth)):
                return [self.evaluationFunction(gameState)]
            #Comprobar si todos los fantasmas han terminado una ronda
            if agent == gameState.getNumAgents() - 1:
                nextAgent=self.index
                depth+= 1
            #Si no pasar al siguiente agente (fantasma)
            else:
                nextAgent = agent + 1

            #Si agente == pacman
            if agent == 0:
                max = -float("inf")
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    newMax = minimax(successor, depth, nextAgent)[0]
                    if newMax >= max:
                        max =newMax
                        bestAction = action
                return [max, bestAction]

            #Si agente == fantasma
            else:
                min = float("inf")
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    newMin = minimax(successor, depth, nextAgent)[0]
                    if newMin <= min:
                        min =newMin
                        bestAction =action
                return [min, bestAction]

        return minimax(gameState,0, self.index)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(gameState, agent, depth, alpha, beta):
            # Comprobar si es un estado terminal o se ha llegado a la profundidad maxima
            bestAction=None
            if (not gameState.getLegalActions(agent) or (depth == self.depth)):
                return [self.evaluationFunction(gameState)]
            # Comprobar si todos los fantasmas han terminado una ronda
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index
                depth += 1
            # Si no pasar al siguiente agente (fantasma)
            else:
                nextAgent = agent + 1

            # Si agente == pacman
            if agent == 0:
                max = -float("inf")
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    newAlpha = alphaBeta(successor, depth, nextAgent,alpha,beta)[0]
                    if newAlpha >= alpha:
                        alpha = newAlpha
                        bestAction = action
                return [alpha, bestAction]

            # Si agente == fantasma
            else:
                min = float("inf")
                for action in gameState.getLegalActions(agent):
                    successor = gameState.generateSuccessor(agent, action)
                    newBeta = alphaBeta(successor, depth, nextAgent,alpha,beta)[0]
                    if newBeta <= beta:
                        beta = newBeta
                        bestAction = action
                return [beta, bestAction]


        return alphaBeta(gameState, self.index, 0, -float("inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, agent, depth):
            # Comprobar si es un estado terminal o se ha llegado a la profundidad maxima
            if (not gameState.getLegalActions(agent) or (depth == self.depth)):
                return [self.evaluationFunction(gameState)]
            # Comprobar si todos los fantasmas han terminado una ronda
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index
                depth += 1
            # Si no, pasar al siguiente agente (fantasma)
            else:
                nextAgent = agent + 1

            bestAction = None
            numChild = len(gameState.getLegalActions(agent))
            value = 0

            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                expecMax = expectimax(successor, nextAgent, depth)[0]
                if agent == self.index:
                    #Calculamos el valor max para el pacman
                    if expecMax > value:
                        value = expecMax
                        bestAction = action
                else:
                    #Calculamos  el valor medio para los fantasmas
                    value = value + (expecMax/ numChild)
            return [value, bestAction]


        return expectimax(gameState, self.index,0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman=currentGameState.getPacmanPosition()
    ghostPositions=currentGameState.getGhostStates()
    ghosts=[]
    scaredGhosts=[]

    #Cargamos los fantasmas
    for ghost in ghostPositions:
        if ghost.scaredTimer:   #Si los fantasmas están asustados
            scaredGhosts.append(ghost)
        else:                   #Si no están asustados
            ghosts.append(ghost)

    #Obtenemos los puntos obtenidos
    #Los puntos nos van a servir para elegir la acción que tomará pacman
    #A cada acción le añadiremos la preferencia relativa
    pt=0
    #Los puntos del estado
    pt+=5*currentGameState.getScore()
    #Sumamos a la puntuación el número de comidas
    pt+=-10*len(currentGameState.getFood().asList())
    #Sumamos a la puntuación el número de capsulas
    pt+=-20*len(currentGameState.getCapsules())

    #Inicializamos las distancias para cada objeto
    foodDistances = []
    ghostsDistances = []
    scaredGhostsDistances = []

    #Obtenemos las distancias hacía la comida
    for food in currentGameState.getFood().asList():
        foodDistances.append(manhattanDistance(pacman, food))

    #Obtenemos las distancias de los fantasmas al pacman
    for ghost in ghosts:
        ghostsDistances.append(manhattanDistance(pacman, ghost.getPosition()))

    #Obtenemos las distancias de los fantasmas asustados al pacman
    for ghost in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacman, ghost.getPosition()))

    #Para tener una mejor puntuación, cuando una comida esta muy cerca conviene comerla
    #ya que puede que al rededor no haya y acabe perdiendo mas tiempo
    for food in foodDistances:
        if food<3:
            pt+=-1*food
        if food<7:
            pt+=-0.75*food
        else:
            pt+=-0.5*food

    #Cuando tenemos fantasmas asustados cerca, conviene comerlos, ya que nos dan muchos puntos y
    #luego es muy probable que el fantasma aparezca alejado a la posición del pacman
    for ghost in scaredGhostsDistances:
        if ghost<3:
            pt+=-30*ghost
        else:
            pt+=-15*ghost

    #Cuando tenemos fantasmas normales cerca, queremos alejarnos de él, por lo que tendrá una
    #importancia relevante
    for ghost in ghostsDistances:
        if ghost<3:
            pt+=5 * ghost
        elif ghost<7:
            pt+=2*ghost
        else:
            pt+=0.5*ghost

    return pt

# Abbreviation
better = betterEvaluationFunction
