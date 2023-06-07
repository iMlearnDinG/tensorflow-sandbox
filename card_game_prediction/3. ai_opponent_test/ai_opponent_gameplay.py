import random
import time
from enum import Enum
from itertools import permutations

import numpy as np
import pygame
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(f'card_game_models/ai_best_model.h5')


class Suit(Enum):
    Spades = 1
    Hearts = 2
    Diamonds = 3
    Clubs = 4


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank} {self.suit.name}"


class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]
        self.discard_pile = []

    def shuffle(self):
        random.seed(time.time())
        random.shuffle(self.cards)

    def deal(self):
        if not self.cards:
            self.cards, self.discard_pile = self.discard_pile, self.cards
            self.shuffle()
        return self.cards.pop()

    def discard(self, card):
        self.discard_pile.append(card)


class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.hand = []

    def add_point(self):
        self.score += 1


class Assets:
    def __init__(self):
        pass

    def draw(self):
        pass


class Score:
    def __init__(self):
        self.players = {"player1": 0, "player2": 0}

    def update_score(self, player):
        self.players[player] += 1

    def get_score(self, player):
        return self.players[player]


def rank_proximity(player_card, dealer_card):
    return abs(player_card.rank - dealer_card.rank) % 13


def suit_proximity(player_card, dealer_card):
    return abs(player_card.suit.value - dealer_card.suit.value)


def compare_cards(player1_card, player2_card, dealer_card):
    rank_diff1 = min(abs(player1_card.rank - dealer_card.rank), abs(13 - abs(player1_card.rank - dealer_card.rank)))
    rank_diff2 = min(abs(player2_card.rank - dealer_card.rank), abs(13 - abs(player2_card.rank - dealer_card.rank)))

    if rank_diff1 < rank_diff2:
        return "player1"
    elif rank_diff1 > rank_diff2:
        return "player2"
    else:
        if player1_card.suit.value > player2_card.suit.value:
            return "player1"
        elif player1_card.suit.value < player2_card.suit.value:
            return "player2"
        else:
            return "tie"


def print_cards(players, dealer):
    print("Player 1 cards:")
    for card in players[0].hand:
        print(card)
    print("\nPlayer 2 cards:")
    for card in players[1].hand:
        print(card)
    print("\nDealer cards:")
    for card in dealer.hand:
        print(card)


def roll_dice():
    return random.randint(1, 6)


def select_card_ai(player1_hand, hand, dealer_cards):
    # Simulate all possible permutations of the AI's hand
    best_perm = None
    best_wins = -1
    for perm in permutations(hand):
        # Convert the game state to the format your model was trained on
        # You'll need to adjust this to match your training data
        X = np.zeros((1, 21))

        X = np.zeros((1, 32))
        # Create a list of card ranks and append zeros until length is 5
        player1_ranks = [card.rank for card in player1_hand] + [0] * (5 - len(player1_hand))
        player1_suits = [card.suit.value for card in player1_hand] + [0] * (5 - len(player1_hand))
        X[0, 0:5] = player1_ranks  # P1 card ranks
        X[0, 5:10] = player1_suits  # P1 card suits

        player2_ranks = [card.rank for card in perm] + [0] * (5 - len(perm))
        player2_suits = [card.suit.value for card in perm] + [0] * (5 - len(perm))
        X[0, 10:15] = player2_ranks  # P2 card ranks
        X[0, 15:20] = player2_suits  # P2 card suits

        dealer_ranks = [card.rank for card in dealer_cards] + [0] * (5 - len(dealer_cards))
        dealer_suits = [card.suit.value for card in dealer_cards] + [0] * (5 - len(dealer_cards))
        X[0, 20:25] = dealer_ranks  # Dealer card ranks
        X[0, 25:30] = dealer_suits  # Dealer card suits

        # Predict the outcome
        y = model.predict(X)
        wins = np.sum(y[0] == "player2")  # Count how many columns player 2 (AI) wins

        # If this permutation results in more wins, update the best permutation
        if wins > best_wins:
            best_perm = perm
            best_wins = wins

    # Update the AI's hand to the best permutation
    hand[:] = best_perm

    # Return the first card in the best permutation
    return hand.pop(0)


def select_card_player(hand):
    while True:
        try:
            card_index = int(input("Choose a card to play (1-5): ")) - 1  # Adjust for zero indexing
            card = hand.pop(card_index)  # Pop removes the card from the hand
            return card
        except (IndexError, ValueError):  # Catch the exceptions when the input is not a valid index
            print("Invalid card, please choose 1 of the available cards.")


def main():
    pygame.init()
    assets = Assets()
    players = [Player("player1"), Player("player2")]
    dealer = Player("dealer")
    score = Score()

    # Initialize and deal cards for the first game
    deck = Deck()
    deck.shuffle()
    for _ in range(5):
        for player in players:
            player.hand.append(deck.deal())
        dealer.hand.append(deck.deal())

    print_cards(players, dealer)

    for i in range(5):
        print("\nColumn", i + 1)

        if len(players[0].hand) == 0:  # If Player 1 has no more cards, end the round
            break

        player1_card = select_card_player(players[0].hand)
        print(f"Player 1 played: {player1_card}")

        player2_card = select_card_ai(players[0].hand, players[1].hand, dealer.hand)
        print(f"Player 2 (AI) played: {player2_card}")

        dealer_card = dealer.hand.pop(0)  # Note that we are popping the card from the dealer's hand
        print(f"Dealer revealed: {dealer_card}")

        winner = compare_cards(player1_card, player2_card, dealer_card)
        print("Winner:", winner)

        if winner == "player1":
            score.update_score("player1")
        elif winner == "player2":
            score.update_score("player2")
        else:
            print("Tie! Rolling dice...")
            player1_dice = roll_dice()
            player2_dice = roll_dice()
            print("Player 1 rolled a", player1_dice)
            print("Player 2 rolled a", player2_dice)

            if player1_dice > player2_dice:
                score.update_score("player1")
            elif player1_dice < player2_dice:
                score.update_score("player2")

        # Print the remaining dealer's hand and columns left
        print("\nDealer's remaining cards:")
        for idx, card in enumerate(dealer.hand, start=1):
            print(f"Column {idx + 1}: {card}")

    # Print the final scores:
    print("Final scores:")
    print("Player 1:", score.get_score("player1"))
    print("Player 2:", score.get_score("player2"))

    pygame.quit()


if __name__ == "__main__":
    main()
