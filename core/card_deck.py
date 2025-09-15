import random
import itertools

# Simplified game classes (keeping these minimal)
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = self._get_value()
    
    def _get_value(self):
        if self.rank == 'A': return 14
        elif self.rank == 'K': return 13
        elif self.rank == 'Q': return 12
        elif self.rank == 'J': return 11
        else: return int(self.rank)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return str(self)

class Deck:
    def __init__(self):
        self.reset()
    
    def reset(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(self.cards)
    
    def draw(self, n=1):
        drawn = []
        for _ in range(n):
            if self.cards:
                drawn.append(self.cards.pop())
            else:
                # Deck is empty - this shouldn't happen in normal poker
                print(f"WARNING: Deck ran out of cards! This should not happen in normal poker.")
                print(f"  Cards already drawn: {52 - len(self.cards)}")
                print(f"  Attempting to draw: {n - len(drawn)} more")
                import traceback
                print("  Stack trace:")
                traceback.print_stack()
                drawn.append(None)
        return drawn if n > 1 else drawn[0] if drawn else None
    
def evaluate_hand(cards):
    """Evaluate poker hand strength for showdown"""
    if len(cards) < 5:
        return 0
    
    best_score = 0
    for combo in itertools.combinations(cards, 5):
        score = score_five_cards(list(combo))
        best_score = max(best_score, score)
    
    return best_score

def score_five_cards(cards):
    """Score a 5-card poker hand"""
    values = sorted([c.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    value_counts = {}
    for v in values:
        value_counts[v] = value_counts.get(v, 0) + 1
    
    counts = sorted(value_counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    
    # Check for straight
    is_straight = False
    if len(set(values)) == 5:
        if values[0] - values[4] == 4:
            is_straight = True
        elif values == [14, 5, 4, 3, 2]:
            is_straight = True
            values = [5, 4, 3, 2, 1]
    
    # Scoring
    if is_flush and is_straight:
        if values[0] == 14:
            return 10000000  # Royal flush
        return 9000000 + values[0] * 10000  # Straight flush
    elif counts == [4, 1]:
        quad = [v for v, c in value_counts.items() if c == 4][0]
        kicker = [v for v, c in value_counts.items() if c == 1][0]
        return 8000000 + quad * 10000 + kicker
    elif counts == [3, 2]:
        trip = [v for v, c in value_counts.items() if c == 3][0]
        pair = [v for v, c in value_counts.items() if c == 2][0]
        return 7000000 + trip * 10000 + pair * 100
    elif is_flush:
        return 6000000 + sum(v * (15 ** (4-i)) for i, v in enumerate(values))
    elif is_straight:
        return 5000000 + values[0] * 10000
    elif counts == [3, 1, 1]:
        trip = [v for v, c in value_counts.items() if c == 3][0]
        kickers = sorted([v for v, c in value_counts.items() if c == 1], reverse=True)
        return 4000000 + trip * 10000 + kickers[0] * 100 + kickers[1]
    elif counts == [2, 2, 1]:
        pairs = sorted([v for v, c in value_counts.items() if c == 2], reverse=True)
        kicker = [v for v, c in value_counts.items() if c == 1][0]
        # FIX: Include the kicker in the score calculation for two pair
        return 3000000 + pairs[0] * 10000 + pairs[1] * 100 + kicker
    elif counts == [2, 1, 1, 1]:
        pair = [v for v, c in value_counts.items() if c == 2][0]
        kickers = sorted([v for v, c in value_counts.items() if c == 1], reverse=True)
        return 2000000 + pair * 10000 + kickers[0] * 100 + kickers[1] * 10 + kickers[2]
    else:
        return 1000000 + sum(v * (15 ** (4-i)) for i, v in enumerate(values))