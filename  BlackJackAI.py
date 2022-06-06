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
            self.score = 0
            self.aces = 0
            
        def takeCard(self, card):
            self.hand.append(card)
            self.score += card.getCardValue()
            if card.rank == "Ace":
                self.aces += 1
            
        def aceNumber(self):
            return self.aces
        
        def handScore(self):
            scr = copy.copy(self.score)
            if self.score > 21 and self.aces > 0:
                scr -= 10 * self.aces
            return scr
            
        def showCard(self):
            self.hand[-1].printCard()
            return self.hand[-1]
            
def playerChoice():
        choice = input("Hit (h) or stand (s)?")
        while choice != "h" and choice != "s":
            choice = input("Hit (h) or stand (s)?")
        return choice == "h"
        
        
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
        
        while len(pockets) > 0:
            for x, pocket in enumerate(pockets):      
                decks.append(Deck())      
                dealers.append(Hand())
                players.append(Hand())
                #print(type(x))
                #print(x)
                dealers[x].takeCard(decks[x].dealCard())
                print(f"Dealer {x} got ")
                dealers[x].showCard()
                dealers[x].takeCard(decks[x].dealCard())
                players[x].takeCard(decks[x].dealCard())
                print(f"Player {x} got ")
                players[x].showCard()
            for x, player in enumerate(players):   
                while player.handScore() < 21:
                    ge[x].fitness += 0.1
                    output = nets[x].activate((
    pockets[x],
    player.handScore(),
    #player.hand,
    dealers[x].hand[0].getCardValue(),
    dealers[x].hand[0].cardRank.index(dealers[x].hand[0].rank)
    ))
                    if output[0] > 0.5:
                        player.takeCard(decks[x].dealCard())
                        print(f"Player{x} got ")
                        player.showCard()
                        ge[x].fitness += 0.5
                    else:
                        break
                if player.handScore() > 21:
                    print(f"Player {x} score is {player.handScore()} and Dealer {x} score is {dealers[x].handScore()}. \nPlayer {x} lost.")
                    ge[x].fitness -= 5
                    pockets[x] -= pockets[x] / 2
                    print(f"Player's {x} total is {pockets[x]}")
                    #break
                elif player.handScore() <= 21:                
                    print(f"Dealer {x} got")
                    dealers[x].showCard()
                    while dealers[x].handScore() < 17:
                        dealers[x].takeCard(decks[x].dealCard())
                        print(f"Dealer {x} got")
                        dealers[x].showCard()
                elif dealers[x].handScore() > 21 or player.handScore() > dealers[x].handScore():
                    print(f"Player's {x}' score is {player.handScore()} and Dealer's {x}' score is {dealers[x].handScore()}. \nPlayer {x} win.")
                    ge[x].fitness += 10
                    pockets[x] += pockets[x] / 2
                    print(f"Player's {x} total is {pockets[x]}")
                    #break
                elif player.handScore() < dealers[x].handScore():
                    print(f"Player {x} score is {player.handScore()} and Dealer {x} score is {dealers[x].handScore()}. \nPlayer {x} lost.")
                    ge[x].fitness -= 5
                    pockets[x] -= pockets[x] / 2
                    print(f"Player's {x} total is {pockets[x]}")
                    #break
                elif player.handScore() == dealers[x].handScore():
                    print(f"Player's {x}' score is {player.handScore()} and Dealer's {x}' score is {dealers[x].handScore()}. \npush")
                    ge[x].fitness += 10
                    pockets[x] += pockets[x] / 2
                    print(f"Player's {x} total is {pockets[x]}")
                    #break
            for x, pocket in enumerate(pockets):   
                if pocket < 15:
                     ge[x].fitness -= 100
                     decks.pop(x)
                     pockets.pop(x)
                     nets.pop(x)
                     ge.pop(x)
                     dealers.pop(x)
                     players.pop(x)
                        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    p = neat.Population(config)
    
    winner = p.run(eval_genomes, 100)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

print("Тут был Леха")