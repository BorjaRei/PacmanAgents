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
        #Obtenemos cuál es la comida más cercana para el pacman
        distancias=[]
        for foodPos in newFood.asList():
            distancias.append(util.manhattanDistance(newPos,foodPos))
        closestFood=min(distancias)
        ptFood=1/closestFood

        #Obtenemos cuál es la mejor opción para el fantasma
        "Primero obtenemos las posiciones de los fantasmas"
        ghostPos=[]
        for pos in newGhostStates:
            "Miramos si el fantastma no está asustado"
            if pos.scaredTimer == 0:
                ghostPos.append(pos)
        closestGhost=min(ghostPos)

        #El fantasma más cercano tomará la decisión que más le convenga
        #TODO: No entiendo que hace aquí
        return successorGameState.getScore()

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
        #Como minimax es recursivo, conviene crear una función para facilitar la recursión
        return self.minimax(gameState,0,1)

    def minimax(self,gameState,depth,agentIndex):
        #Guardamos los movimientos posibles
        movements = []
        for action in gameState.getLegalActions(depth):
            if action != 'Stop':
                movements.append(action)

        # Actualizamos la profundidad
        agentIndex+=1
        if agentIndex >= gameState.getNumAgents():
            agentIndex=0
            depth+=1

        #Elegimos el mejor resultado
        results=[]
        for action in movements:
            results.append(self.minimax(gameState.generateSuccessor(agentIndex,action),depth,agentIndex))

        if agentIndex==0:   #Turno del pacman
            bestResult=max(results)
        else:               #Turno del fantasma
            bestResult=min(results)
        return bestResult


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

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
