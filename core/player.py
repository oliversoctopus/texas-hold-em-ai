class Player:
    def __init__(self, name, chips=1000, is_ai=False):
        self.name = name
        self.chips = float(chips)
        self.hand = []
        self.is_ai = is_ai
        self.folded = False
        self.current_bet = 0
        self.all_in = False
        self.ai_model = None
        self.total_invested = 0  # Track total investment in pot
    
    def reset_hand(self):
        self.hand = []
        self.folded = False
        self.current_bet = 0
        self.all_in = False
        self.total_invested = 0
    
    def bet(self, amount):
        # Round to avoid floating point errors
        amount = round(amount, 2)
        actual = min(amount, self.chips)
        actual = round(actual, 2)
        self.chips = round(self.chips - actual, 2)
        self.current_bet = round(self.current_bet + actual, 2)
        self.total_invested = round(self.total_invested + actual, 2)
        if self.chips == 0:
            self.all_in = True
        return actual