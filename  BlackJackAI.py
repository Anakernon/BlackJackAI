import random
from random import shuffle
import os
import neat
import copy

class Card:
    
    cardRank = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")
    cardSuit = ("Club", "Diamond", "Heart", "Spade")
    
    def __init__(self, rank = cardRank[0], suit = cardSuit[0]):
        self.rank = rank
        self.suit = suit
        
    def printCard(self):
        print(f"{self.rank} {self.suit}")
        
    def getCardValue(self):
        self.cardValue = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)
        self.cVs = dict(zip(self.cardRank, self.cardValue))
        return self.cVs[self.rank]
        
        
class Deck:
    
    def __init__(self):
        self.deck = []
        for rank in Card().cardRank:
            for suit in Card().cardSuit:
                self.deck.append(Card(rank, suit))
        random.shuffle(self.deck)
        
        
    def dealCard(self):
        return self.deck.pop(0)
        
class Hand:
        
        def __init__(self):
            self.hand = []
            
        def takeCard(self, card):
            self.hand.append(card)
            
        def handScore(self):
            aces = 0
            score = 0
            for card in self.hand:
                score += card.getCardValue()
                if "Ace" == card.rank:
                    aces += 1
            if score > 21 and aces > 0:
                score -= 10 * aces
            return score
            
        def showCard(self):
            self.hand[-1].printCard()
            return self.hand[-1]
            
        def aceNumber(self):
            aces = 0
            for card in self.hand:
                if "Ace" == card.rank:
                    aces += 1
            return aces
          
def eval_genomes(genomes, config):
        decks = []
        pockets = []
        nets = []
        ge = []
        dealers = []
        players = []
        for _, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            ge.append(genome)
            pockets.append(100)
            decks.append(Deck())
            dealers.append(Hand())
            players.append(Hand())
        
        while len(pockets) > 0:
            for x, pocket in enumerate(pockets):      
                decks[x] = Deck()
                dealers[x] = Hand()
                players[x] = Hand()
                dealers[x].takeCard(decks[x].dealCard())
                #print(f"Dealer{x} got ")
                #dealers[x].showCard()
                dealers[x].takeCard(decks[x].dealCard())
                players[x].takeCard(decks[x].dealCard())
                #print(f"Player{x} got ")
                #players[x].showCard()
                while players[x].handScore() < 21:
                    ge[x].fitness += 0.05
                    output = nets[x].activate((
    pockets[x],
    players[x].handScore(),
    #player.hand,
    dealers[x].hand[0].getCardValue(),
    dealers[x].hand[0].cardRank.index(dealers[x].hand[0].rank)
    ))
                    if output[0] > 0.5:
                        players[x].takeCard(decks[x].dealCard())
                        #print(f"Player{x} got ")
                        #players[x].showCard()
                        ge[x].fitness += 0.005
                    else:
                        break
                if players[x].handScore() > 21:
                    #print(f"Player{x} score is {players[x].handScore()} and Dealer{x} score is {dealers[x].handScore()}. \nPlayer{x} lost.")
                    ge[x].fitness -= 0.05
                    pockets[x] -= 50
                    print(f"Player{x} total is {pockets[x]}")
                else:                
                    #print(f"Dealer{x} got")
                    #dealers[x].showCard()
                    while dealers[x].handScore() < 17:
                        dealers[x].takeCard(decks[x].dealCard())
                        #print(f"Dealer{x} got")
                        #dealers[x].showCard()
                    if dealers[x].handScore() > 21 or players[x].handScore() > dealers[x].handScore():
                        #print(f"Player{x}' score is {players[x].handScore()} and Dealer{x}' score is {dealers[x].handScore()}. \nPlayer{x} win.")
                        ge[x].fitness += 0.5
                        pockets[x] += 50
                        print(f"Player{x} total is {pockets[x]}")
                    elif players[x].handScore() < dealers[x].handScore():
                        #print(f"Player{x} score is {players[x].handScore()} and Dealer{x} score is {dealers[x].handScore()}. \nPlayer{x} lost.")
                        ge[x].fitness -= 0.05
                        pockets[x] -= 50
                        print(f"Player{x} total is {pockets[x]}")
                    elif players[x].handScore() == dealers[x].handScore():
                        #print(f"Player{x}' score is {players[x].handScore()} and Dealer{x}' score is {dealers[x].handScore()}. \npush")
                        ge[x].fitness += 0.15
                        pockets[x] += 50
                        print(f"Player{x} total is {pockets[x]}")
                if pocket < 15:
                     ge[x].fitness -= 0.75
                     decks.pop(x)
                     pockets.pop(x)
                     nets.pop(x)
                     ge.pop(x)
                     dealers.pop(x)
                     players.pop(x)
                        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 150)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

print("Тут был Леха")