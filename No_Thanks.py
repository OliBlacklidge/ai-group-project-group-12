import random
import matplotlib as plt

# 1. Deck
# ----------------------------------------------------------------------------

class Deck(object):
    """
    Deck consists of list of numbers (cards). Is initialised with standard list
    of cards in No Thanks!. Decks can be shuffled, drawn from and number of 
    cards counted.
    """
    
    def __init__(self):
        self.deck = []
        
    def build(self, verbose):
        cards_all = range(3,36)
        deck = random.sample(cards_all, 24)
        
        for card in deck:
            self.deck.append(card)
        
        if verbose:
            print("The deck has been shuffled.")
            
    def draw(self):
        return self.deck.pop()

    def check_end(self):
        if self.deck == []:
            return True
        
# 2. Player
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
        self.chip_hand = 11
        
    def draw_card(self, deck, player, verbose):
        global card_pool
        global turn_no
        
        card_pool = deck.draw()
        if verbose:
            print(f'{self.name} draws the number ' + str(card_pool) + ".")
        
        player.player_strat(player, deck, turn_no, verbose)
    
    def take_card(self, player, deck, verbose):
        global card_pool
        global chip_pool
        global game_end
        
        self.card_hand.append(card_pool)
        self.chip_hand += chip_pool
        
        if verbose:
            print(f'{self.name} takes the ' + str(card_pool) + " and " + str(chip_pool) + " chips.")
            print(f'{self.name} has ' + str(self.chip_hand) + ' chips remaining.')
        
        chip_pool = 0
        
        if not deck.check_end():
            player.draw_card(deck, player, verbose)
        else:
            game_end = True
        
    def pass_card(self, verbose):
        
        global card_pool
        global chip_pool
        
        self.chip_hand -= 1
        chip_pool += 1
        
        if verbose:
            print(f'{self.name} passes the ' + str(card_pool) + " and loses a chip.")
            print(f'{self.name} has ' + str(self.chip_hand) + ' chips remaining.')
     
    def player_strat(self, player, deck, turn_no, verbose):
        
        if turn_no % 3 == 1:
            player.weighted_play(player, deck, verbose)
            
        if turn_no % 3 == 2:
            player.rand_play(player, deck, verbose)
            
        if turn_no % 3 == 0:
            player.rand_play(player, deck, verbose)

        
    def rand_play(self, player, deck, verbose):
        """
        Action is randomly determined. Note that players must take a card if
        they are out of chips.
        """
        global chip_pool
        decision = random.randint(0,3)
        
        if self.chip_hand == 0:
            decision = 0

        if decision == 0:
            player.take_card(player, deck, verbose)
            
        else:
            player.pass_card(verbose)
            
    def remove_runs(player_hand):
        player_hand.sort()
        player_hand.reverse()
        remove_list = []
        
        for i in player_hand:
            if i-1 in player_hand:
                remove_list.append(i)

        for i in remove_list:
            player_hand.remove(i)
            
        return player_hand
    
    def point_tally(self):
        self.card_hand = Player.remove_runs(self.card_hand)
        card_points = sum(self.card_hand)
        chip_points = self.chip_hand
        return card_points - chip_points
    
    def chip_weight(chip_count):
        # Linear function mx + c such that chip_weight = 1 when chip_count = 1
        # and chip_weight = 3 when chip_count = 11
        return 0.2 * chip_count + 0.8
        
    
    def weighted_play(self, player, deck, verbose):
        global card_pool
        global chip_pool
        
        take_card_hand = Player.remove_runs(self.card_hand + [card_pool])
        take_chip_hand = self.chip_hand + chip_pool
        pass_card_hand = Player.remove_runs(self.card_hand)
        pass_chip_hand = self.chip_hand - 1
        
        take_value = sum(take_card_hand) - (Player.chip_weight(take_chip_hand)/2) * take_chip_hand
        pass_value = sum(pass_card_hand) - Player.chip_weight(pass_chip_hand) * pass_chip_hand
        
        # print('take_value is '+str(take_value)+' and pass_value is '+str(pass_value))
        
        
        if take_value <= pass_value or self.chip_hand <=0:
            player.take_card(player, deck, verbose)
            
        else:
            player.pass_card(verbose)
            
        
        
        
        
        
    
# 3. Game
# ----------------------------------------------------------------------------

def Run_Game(player_1, player_2, player_3, verbose=False):
    """
    A game reflects an iteration of turns, until the deck emtpies and total
    points are tallied. Winner is then determined. Initialised with three
    players.
    """

    Player_1 = Player(player_1)
    Player_2 = Player(player_2)
    Player_3 = Player(player_3)

    deck = Deck()
    deck.build(verbose)
    global turn_no
    global card_pool
    global chip_pool 
    global game_end
    """
    Global used as card_pool and chip_pool need to be updated each turn so
    cannot be reset between function calls.
    """
    turn_no = 1
    card_pool = 0
    chip_pool = 0
    game_end = False
    random_start = True
    if random_start:
        turn_no = random.randint(1, 3)
        if turn_no == 1:
            Player_1.draw_card(deck, Player_1, verbose)
        elif turn_no == 2:
            Player_2.draw_card(deck, Player_2, verbose)
        elif turn_no == 3:
            Player_3.draw_card(deck, Player_3, verbose)
    
    else:
        turn_no = 1
        Player_1.draw_card(deck, Player_1, verbose)
    
    while not game_end:
        turn_no += 1
        
        if turn_no % 3 == 1:
            Player_1.player_strat(Player_1, deck, turn_no, verbose)
            
        if turn_no % 3 == 2:
            Player_2.player_strat(Player_2, deck, turn_no, verbose)
            
        if turn_no % 3 == 0:
            Player_3.player_strat(Player_3, deck, turn_no, verbose)
            
    else:
        P1_total = Player_1.point_tally()
        P2_total = Player_2.point_tally()
        P3_total = Player_3.point_tally()
        
        if verbose:
            print(f'{Player_1.name} has a final score of ' + str(P1_total))
            print(f'{Player_2.name} has a final score of ' + str(P2_total))
            print(f'{Player_3.name} has a final score of ' + str(P3_total))
        
        if min(P1_total, P2_total, P3_total) == P1_total:
            if verbose:
                print(f'{Player_1.name} has won!!!')
            return 1
            
        elif min(P1_total, P2_total, P3_total) == P2_total:
            if verbose: 
                print(f'{Player_2.name} has won!!!')
            return 0
         
        elif min(P1_total, P2_total, P3_total) == P3_total:
            if verbose:
                print(f'{Player_3.name} has won!!!')
            return 0



weighted_win_proportion = 0            
for i in range(0, 10000):
    weighted_win_proportion += Run_Game('Alice', 'Bob', 'Charlie')

weighted_win_proportion /= 100
print('Alice won '+str(weighted_win_proportion)+'% of the  time.')


#Run_Game('Alice', 'Bob', 'Charlie', True)
