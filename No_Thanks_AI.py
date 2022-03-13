import random
import numpy as np
import pandas as pd
import itertools

# 1. State, Action and Reward
# ----------------------------------------------------------------------------
    
def states():
    """
    Computes every possible state.
    
    Each state consists of 4 elements:
        
        state[0] = the open card (the card that is currently being passed between players)
        state[1] = the number of chips on the open card
        state[2] = the number of chips currently in possession of the player
        state[3] = the list of cards currently in possession of the player
    """
    
    open_card_states = [-x for x in range(3,12)]
    open_chip_states = [x for x in range(13)]
    player_chip_states = [x for x in range(13)]
    
    player_cards = [-i for i in range(3,12)]
    possible_combos = []
    for i in range(7):
        possible_combos.append(list(itertools.combinations(player_cards,i)))
    
    S = [[i,j,k] for i in player_chip_states for j in open_chip_states for k in open_card_states]
    combos = []
    
    for val in S:
        for i in range(len(possible_combos)):
            for j in range(len(possible_combos[i])):
                List = list(possible_combos[i][j])
                while len(List) < 6:
                    List.insert(0,0)
                for n in val:
                    List.insert(0,n)
                combos.append(List)
                
    states = list()
    
    for i in range(len(combos)):
        if (combos[i][1] + combos[i][2] <= 44) and (combos[i][0] in combos[i][3:9]) == False:
            states.append(combos[i])
            
    return states

def state_dict(states = states()):

    state_dict = dict()
    
    for val in range(len(states)):
        state_dict[str(states[val])] = val
    
    return state_dict
    
def actions():
    """
    Computes every possible action available to the player.
    
    For No Thanks!, there's only two possible actions: take or pass.
    """
    
    actions_all = ["take", "pass"]
    
    return actions_all

def card_point_tally(hand):
    """
    Calculates the total sum of all cards in a players hand, taking into account 
    that only the lowest value card counts in a run of consecutive cards.
    """
    
    hand.sort()
    
    i = 1
    while i < len(hand):
        if hand[i] == hand[i-1] + 1:
            hand.remove(hand[i-1])
        else:
            i += 1
    
    return sum(hand)
    
def rewards(states, actions, state_dict = state_dict()):
    """
    Initialises the reward matrix. Each value corresponds to the total points of the player 
    if action[i] is chosen at state[i].
    """
    
    R = np.zeros((len(states), len(actions)))
    
    for i in range(len(states)):
        R[i][0] = states[i][2] + states[i][1] - card_point_tally([val for idx, val in enumerate(states[i]) if idx not in [1,2]])
        R[i][1] = states[i][2] - card_point_tally(states[i][3:12]) - 1
    
    R = pd.DataFrame(data = R, columns = actions, index = state_dict.values())
    
    return R

# 2. Agents
# ----------------------------------------------------------------------------
    
class MonteCarloAgent(object):
    """
    Given the discrete state-action matrix, the agent navigates through the 
    fields by simulating multiple games. While the matrix is initialized with 
    all values at zero, Monte Carlo (MC) updates all visited state-action 
    values after every completed game.
    
    q(s,a) = q(s,a) + (alpha) * (R - q(s,a))
    The q-value at state s taking action a is updated dependent on the 
    achieved reward in this episode R, and the step size parameter alpha. In 
    order to decide which action to take in a respective state, the 
    epsilon-greedy form of the algorithm chooses:
    With epsilon probability: Random action
    With 1-epsilon probability: Action with maximum q-values
    """
    
    def agent_init(self, agent_init_info):
        """
        Initializes the agent to get parameters and import/create q-tables.
        Required parameters: agent_init_info as dict
        """
        
        # (1) Store the parameters provided in agent_init_info
        self.states = states()
        self.state_dict = state_dict()
        self.actions = actions()
        self.state_seen = list()
        self.action_seen = list()
        self.q_seen = list()
        
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.R = rewards(self.states, self.actions)
        
        self.q = pd.DataFrame(data = np.zeros((len(self.states),len(self.actions))), columns = self.actions, index = self.state_dict.values())
        self.visit = self.q.copy()
        
    def step(self, state_dict, actions_dict):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        """
        
        # (1) Transform state dictionary into tuple
        state = self.state_dict[str(state_dict)]
        
        # (2) Choose action using epsilon greedy
        # (2a) Random action
        if random.random() < self.epsilon:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            action = random.choice(actions_possible)
         
        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0
            
            for i in actions_possible:
                val = self.q.loc[[state],i][0]
                if val >= val_max:
                    val_max = val
                    action = i
        
        # (3) Add state-action pair if not seen in this simulation
        if ((state),action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
            
        self.q_seen.append(((state),action))
        self.visit.loc[[state], action] += 1
        
        return action
    
    def update(self, state_dict, action):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str
        """
        
        state = [i for i in state_dict.values()]
        state = tuple(state)
        reward = self.R.loc[[state], action][0]
        
        # Update Q-values of all state-action pairs visited in the simulation
        for s,a in zip(self.state_seen, self.action_seen):
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
            print(self.q.loc[[s],a])
            
        self.state_seen, self.action_seen, self.q_seen = list(), list(), list()
        
        
class QLearningAgent(object):
    """
    In its basic form, Q-learning works in a similar way. However, while MC 
    waits for the completion of each episode before updating q-values, 
    Q-learning updates them with a lag of one step, at each step.
    
    q(s,a) = q(s,a) + (alpha) * (r + q(s-hat,a-hat) - q(s,a))
    
    The q-value is thereby dependent on the step-size parameter, the reward of 
    the next step r, and the q-value of the next step at state s-hat and 
    action-hat.
    Both algorithms consequently take the same 2 parameters which have the 
    following effects:
    Alpha: A higher step size parameter increases the change in q-values at 
    each update while prohibiting values to converge closer to their true 
    optimum.
    
    Epsilon: A higher epsilon grants more exploration of actions, which do not
    appear profitable at first sight. At the same time, this dilutes the 
    optimal game strategy when it has been picked up by the agent.
    """
    
    def agent_init(self, agent_init_info):
        """
        Initializes the agent to get parameters and import/create q-tables.
        Required parameters: agent_init_info as dict
        """
        global q
        
        # (1) Store the parameters provided in agent_init_info
        self.states = states()
        self.state_dict = state_dict()
        self.actions = actions()
        self.prev_state = 0
        self.prev_action = 0
        
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.R = rewards(self.states, self.actions)
        
        self.q = pd.DataFrame(data = np.zeros((len(self.states),len(self.actions))), columns = self.actions, index = self.state_dict.values())
        self.visit = self.q.copy()
        
    def step(self, state_dict, actions_dict):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        """
        
        # (1) Transform state dictionary into tuple
        state = self.state_dict[str(state_dict)]
        
        # (2) Choose action using epsilon greedy
        # (2a) Random action
        if random.random() < self.epsilon:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            action = random.choice(actions_possible)
         
        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0
            
            for i in actions_possible:
                val = self.q.loc[state,i]
                if val >= val_max:
                    val_max = val
                    action = i
        
        return action
    
    def update(self, state_dict, action, i):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str
        """
        state = self.state_dict[str(state_dict)]
        
        # (1) Set prev_state unless first turn
        if self.prev_state != 0:
            prev_q = self.q.loc[self.prev_state, self.prev_action]
            this_q = self.q.loc[state, action]
            reward = self.R.loc[state, action]
            
            # Calculate new Q-values
            if reward == 0:
                self.q.loc[self.prev_state, self.prev_action] = prev_q + self.step_size * (reward + this_q - prev_q)
            else:
                self.q.loc[self.prev_state, self.prev_action] = prev_q + self.step_size * (reward - prev_q)
        
            self.visit.loc[self.prev_state, self.prev_action] += 1
            
        
        # (2) Save and return action/state
        self.prev_state = self.state_dict[str(state_dict)]
        self.prev_action = action
        
        if i % 100 == 0:
            self.q.to_csv('C:/Users/oblac/Documents/GitHub/ai-group-project-group-12/q-tables/q-table itr '+str(i)+'.csv')

# 3. Deck
# ----------------------------------------------------------------------------

class Deck(object):
    """
    Deck consists of list of numbers (cards). Is initialised with cards labelled from 3-11. 
    Decks can be shuffled, drawn from and number of cards counted.
    """
    
    def __init__(self):
        self.deck = []
        
    def build(self):
        cards_all = range(3,12)
        deck = random.sample(cards_all, 6)
        
        for card in deck:
            self.deck.append(card)
            
    def draw(self):
        return self.deck.pop()

    def check_end(self):
        if self.deck == []:
            return True
        
# 4. Player
# ----------------------------------------------------------------------------
        
class Player(object):
    """
    Player consists of a list of cards and number of chips in posession.
    Players can take or pass cards and/or chips. Total points to player at any
    time can be calculated.
    """
    
    def __init__(self, name):
        self.name = name
        self.card_hand = list()
        self.chip_hand = 3
        
        self.state = list()
        self.actions = dict()

    def draw_card_rand(self, deck, player):
        global card_pool
        
        card_pool = deck.draw()
        # print(f'{self.name} draws the number ' + str(card_pool) + ".")
        
        player.rand_play(player, deck)
        
    def draw_card_agent(self, deck, player, i):
        global card_pool
        
        card_pool = deck.draw()
        # print(f'{self.name} draws the number ' + str(card_pool) + ".")
        
        player.play_agent(player, deck, i)
    
    def take_card_rand(self, player, deck):
        global card_pool
        global chip_pool
        global game_end
        
        self.card_hand.append(-card_pool)
        self.card_hand.sort(reverse=True)
        self.chip_hand += chip_pool
        
        chip_pool = 0
        
        if not deck.check_end():
            player.draw_card_rand(deck, player)
        else:
            game_end = True
            
    def take_card_agent(self, player, deck, i):
        global card_pool
        global chip_pool
        global game_end
        
        self.card_hand.append(-card_pool)
        self.card_hand.sort(reverse=True)
        
        while len(self.card_hand) > 6:
            self.card_hand.remove(self.card_hand[0])
        self.chip_hand += chip_pool
        
        chip_pool = 0
        
        if algorithm == "q-learning":
            agent.update(self.state, self.action, i)
        
        if not deck.check_end():
            player.draw_card_agent(deck, player, i)
        else:
            game_end = True
        
    def pass_card(self):
        global card_pool
        global chip_pool
        
        self.chip_hand -= 1
        chip_pool += 1
        
    def identify_state(self):
    
        while len(self.card_hand) < 6:
            self.card_hand.insert(0,0)
            
        state = [-card_pool, chip_pool, self.chip_hand, self.card_hand]
        
        self.state = []
        for i in state:
            if type(i) == list:
                for j in i:
                    self.state.append(j)
            else:
                self.state.append(i)
        
    def identify_action(self):

        if self.chip_hand == 0:
            self.actions = {"take":1,"pass":0}
        else:
            self.actions = {"take":1,"pass":1}
        
    def play_agent(self, player, deck, i):
        
        self.identify_state()
        self.identify_action()
        
        self.action = agent.step(self.state, self.actions)
        
        if self.action == "take":
            # print("Take")
            player.take_card_agent(player, deck, i)
            
        else:  
            if algorithm == "q-learning":
                agent.update(self.state, self.action, i)
            
            # print("Pass")
            player.pass_card() 
        
    def rand_play(self, player, deck):
        """
        Action is randomly determined. Note that players must take a card if
        they are out of chips.
        """
        global chip_pool
        decision = random.randint(0,1)
        
        if self.chip_hand == 0:
            decision = 0
        
        if decision == 0:
            player.take_card_rand(player, deck)
            
        if decision == 1:
            player.pass_card()
            
    def point_tally(self):
        """
        Calculates the total number of points a player currently has.
        """
        
        card_points = card_point_tally(self.card_hand)
        chip_points = self.chip_hand
        return card_points - chip_points
    
    
# 5. Game
# ----------------------------------------------------------------------------

class Game(object):
    
   def __init__(self, player_1, player_2, player_3, i):
        """
        A game reflects an iteration of turns, until the deck emtpies and total
        points are tallied. Winner is then determined. Initialised with three
        players.
        """
    
        self.player_1 = Player(player_1)
        self.player_2 = Player(player_2)
        self.player_3 = Player(player_3)
    
        deck = Deck()
        deck.build()
        turn_no = 1
        global card_pool
        global chip_pool 
        global game_end
        """
        Global used as card_pool and chip_pool need to be updated each turn so
        cannot be reset between function calls.
        """
        card_pool = 0
        chip_pool = 0
        game_end = False
        random_start = True
        if random_start:
            turn_no = random.randint(1, 3)
            if turn_no == 1:
                self.player_1.draw_card_agent(deck, self.player_1, i)
            elif turn_no == 2:
                self.player_2.draw_card_agent(deck, self.player_2, i)
            elif turn_no == 3:
                self.player_3.draw_card_agent(deck, self.player_3, i)
        
        else:
            turn_no = 1
            self.player_1.draw_card_rand(deck, self.player_1, i)
        
               
        while not game_end:
            turn_no += 1
            
            if turn_no % 3 == 1:
                self.player_1.play_agent(self.player_1, deck, i)
                
            if turn_no % 3 == 2:
                self.player_2.play_agent(self.player_2, deck, i)
                
            if turn_no % 3 == 0:
                self.player_3.play_agent(self.player_3, deck, i)
                
        else:
            P1_total = self.player_1.point_tally()
            P2_total = self.player_2.point_tally()
            P3_total = self.player_3.point_tally()
    
            if max(P1_total, P2_total, P3_total) == P1_total:
                tally[0] += 1
                
            elif max(P1_total, P2_total, P3_total) == P2_total:
                 tally[1] += 1
             
            elif max(P1_total, P2_total, P3_total) == P3_total:
                 tally[2] += 1
                 
                 
def Tournament(player_1, player_2, player_3, match_no, algo, agent_info):
    
    global agent, algorithm, tally
    
    tally = [0,0,0]
    
    algorithm = algo
    
    if algo == "q-learning":
        agent = QLearningAgent()
    else:
        agent = MonteCarloAgent()
        
    agent.agent_init(agent_info)
    
    for i in range(1, match_no+1):
        Game(player_1, player_2, player_3, i)
        
    print(tally)
             
agent_init_info = {"epsilon":0.9, "step_size":0.2, "new_model":True}
            
Tournament('Alice', 'Bob', 'Charlie', 1000, "q-learning", agent_init_info)