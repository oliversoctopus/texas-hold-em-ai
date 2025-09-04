class Player:
    def __init__(self, name, chips=1000, is_ai=False):
        self.name = name
        self.chips = chips
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
        actual = min(amount, self.chips)
        self.chips -= actual
        self.current_bet += actual
        self.total_invested += actual
        if self.chips == 0:
            self.all_in = True
        return actual