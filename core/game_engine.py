import random
from .game_constants import Action
from .card_deck import Card, Deck, evaluate_hand
from .player import Player

class TexasHoldEmTraining:
    """Specialized class for training with better state tracking"""
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.reset_game()
    
    def reset_game(self):
        self.deck = Deck()
        self.players = []
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.action_history = []
        self.opponent_bets = []
        self.dealer_position = random.randint(0, self.num_players - 1)
        self.last_raise_amount = 20  # Initialize to big blind
    
    def simulate_hand(self, ai_models):
        """Simulate a hand and collect experiences"""
        experiences = []
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.action_history = []
        self.opponent_bets = []
        
        # Setup players
        self.players = []
        for i in range(self.num_players):
            player = Player(f"AI_{i}", 1000, is_ai=True)
            player.ai_model = ai_models[i % len(ai_models)]
            self.players.append(player)
            player.reset_hand()
        
        # Deal cards
        for player in self.players:
            if player.chips > 0:
                player.hand = self.deck.draw(2)
        
        # Post blinds
        sb_pos = (self.dealer_position + 1) % len(self.players)
        bb_pos = (self.dealer_position + 2) % len(self.players)
        self.pot += self.players[sb_pos].bet(10)
        self.pot += self.players[bb_pos].bet(20)
        self.current_bet = 20
        
        # Play through streets
        streets = ['preflop', 'flop', 'turn', 'river']
        for street_idx, street in enumerate(streets):
            if street == 'flop':
                # Draw 3 cards and filter out None values
                flop_cards = self.deck.draw(3)
                self.community_cards.extend([c for c in flop_cards if c is not None])
            elif street == 'turn':
                card = self.deck.draw()
                if card:
                    self.community_cards.append(card)
            elif street == 'river':
                card = self.deck.draw()
                if card:
                    self.community_cards.append(card)
            
            # Betting round
            street_experiences = self.betting_round_with_tracking(ai_models, street_idx)
            experiences.extend(street_experiences)
            
            # Check if hand is over
            active = [p for p in self.players if not p.folded]
            if len(active) <= 1:
                break
        
        # Determine winners and calculate final rewards
        winners = self.determine_winners()
        
        # Calculate rewards for all experiences
        for exp_data in experiences:
            player, state, action, next_state = exp_data
            
            # Calculate reward based on outcome
            if player in winners:
                # Winner gets positive reward based on pot won
                reward = (self.pot / len(winners) - player.total_invested) / 100
            elif player.folded:
                # Folding penalty based on pot size and investment
                reward = -player.total_invested / 100 - 0.1
            else:
                # Loser penalty
                reward = -player.total_invested / 100
            
            # Add shaping rewards
            if action == Action.FOLD and player.total_invested < 30:
                reward += 0.05  # Small bonus for folding early with bad hands
            elif action == Action.RAISE and player in winners:
                reward += 0.1  # Bonus for aggressive play when winning
            
            # Store with priority based on reward magnitude
            priority = abs(reward) + 0.1
            player.ai_model.remember(state, action, reward, next_state, True, priority=priority)
        
        return winners
    
    def betting_round_with_tracking(self, ai_models, street_idx):
        """Betting round with proper raise rules and experience tracking"""
        experiences = []
        first_to_act = (self.dealer_position + 3) % len(self.players) if street_idx == 0 else (self.dealer_position + 1) % len(self.players)
        current = first_to_act
        players_acted = set()
        
        # Store current street for execute_action
        self.current_street = street_idx
        
        # Reset minimum raise for new street (keep existing value if mid-street)
        if not hasattr(self, 'last_raise_amount'):
            self.last_raise_amount = 20  # Big blind
        
        for _ in range(len(self.players) * 3):  # Max iterations
            player = self.players[current]
            
            if player.folded or player.all_in or player.chips == 0:
                current = (current + 1) % len(self.players)
                if current == first_to_act:
                    break
                continue
            
            if player.current_bet >= self.current_bet and player in players_acted:
                current = (current + 1) % len(self.players)
                if current == first_to_act:
                    break
                continue
            
            # Check if this is a CFR or Random AI (they don't use state features)
            is_cfr_or_random = (hasattr(player.ai_model, '__class__') and 
                               ('CFRWrapper' in player.ai_model.__class__.__name__ or 
                                'RandomAI' in player.ai_model.__class__.__name__))
            
            if is_cfr_or_random:
                # CFR and Random AIs don't use state features, create a dummy state object
                class DummyState:
                    def __init__(self, player, game):
                        self.hole_cards = player.hand
                        self.community_cards = game.community_cards
                        self.pot_size = game.pot
                        self.to_call = max(0, game.current_bet - player.current_bet)
                        self.stack_size = player.chips
                        self.position = game.players.index(player)
                        self.num_players = game.num_players
                        self.action_history = game.action_history[-10:]
                
                state = DummyState(player, self)
            else:
                # Regular AI with state features
                state = player.ai_model.get_state_features(
                    player.hand, self.community_cards, self.pot, self.current_bet,
                    player.chips, player.current_bet, self.num_players,
                    sum(1 for p in self.players if not p.folded),
                    current, self.action_history[-10:], self.opponent_bets[-10:],
                    hand_phase=street_idx
                )
            
            # Get valid actions and choose
            valid_actions = self.get_valid_actions(player)
            action = player.ai_model.choose_action(state, valid_actions, training=True)
            
            # Execute action
            old_pot = self.pot
            self.execute_action(player, action)
            
            # Get next state
            if is_cfr_or_random:
                next_state = DummyState(player, self)
            else:
                next_state = player.ai_model.get_state_features(
                    player.hand, self.community_cards, self.pot, self.current_bet,
                    player.chips, player.current_bet, self.num_players,
                    sum(1 for p in self.players if not p.folded),
                    current, self.action_history[-10:], self.opponent_bets[-10:],
                    hand_phase=street_idx
                )
            
            # Store experience
            experiences.append((player, state, action, next_state))
            
            self.action_history.append(action)
            
            # Update players_acted based on action
            if action in [Action.RAISE, Action.ALL_IN] and player.current_bet > old_pot:
                players_acted = {player}  # Reset when someone raises
            else:
                players_acted.add(player)
            
            # Check for end conditions
            active = [p for p in self.players if not p.folded and not p.all_in and p.chips > 0]
            if len(active) <= 1 or sum(1 for p in self.players if not p.folded) <= 1:
                break
            
            current = (current + 1) % len(self.players)
            
            # All players acted and matched
            if all(p in players_acted for p in active):
                if all(p.current_bet == self.current_bet for p in active):
                    break
        
        # Reset current bets for next street
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        
        return experiences
    
    def get_valid_actions(self, player):
        """Get valid actions for a player"""
        actions = []
        call_amount = self.current_bet - player.current_bet
        
        if player.chips == 0:
            return [Action.CHECK]
        
        actions.append(Action.FOLD)
        
        if call_amount == 0:
            actions.append(Action.CHECK)
        elif call_amount < player.chips:
            actions.append(Action.CALL)
        
        if player.chips > call_amount * 2:
            actions.append(Action.RAISE)
        
        if player.chips > 0:
            actions.append(Action.ALL_IN)
        
        return actions
    
    def execute_action(self, player, action):
        """Execute a player action with intelligent raise sizing"""
        if action == Action.FOLD:
            player.folded = True
        elif action == Action.CHECK:
            # Validate that checking is actually allowed
            call_amount = self.current_bet - player.current_bet
            if call_amount > 0:
                # Invalid check - convert to fold
                player.folded = True
                return  # Exit early since we converted to fold
            pass
        elif action == Action.CALL:
            amount = player.bet(self.current_bet - player.current_bet)
            self.pot += amount
            self.opponent_bets.append(amount)
        elif action == Action.RAISE:
            # Use intelligent raise sizing for AI
            call_amount = self.current_bet - player.current_bet
            pot_size = self.pot + call_amount
            
            # Determine minimum raise
            if hasattr(self, 'last_raise_amount'):
                min_raise = self.last_raise_amount
            else:
                min_raise = 20  # Big blind
            
            # Get strategic raise size from AI if available
            if player.ai_model and hasattr(player.ai_model, 'get_raise_size'):
                # Check if this is a CFR or Random AI
                is_cfr_or_random = (hasattr(player.ai_model, '__class__') and 
                                   ('CFRWrapper' in player.ai_model.__class__.__name__ or 
                                    'RandomAI' in player.ai_model.__class__.__name__))
                
                if is_cfr_or_random:
                    # CFR and Random AIs use different state format
                    class DummyState:
                        def __init__(self, player, game):
                            self.hole_cards = player.hand
                            self.community_cards = game.community_cards
                            self.pot_size = game.pot
                            self.to_call = max(0, game.current_bet - player.current_bet)
                            self.stack_size = player.chips
                            self.position = game.players.index(player)
                            self.num_players = game.num_players
                            self.action_history = game.action_history[-10:]
                    
                    state = DummyState(player, self)
                else:
                    # Regular AI with state features
                    street_idx = self.current_street if hasattr(self, 'current_street') else 0
                    state = player.ai_model.get_state_features(
                        player.hand, self.community_cards, self.pot, self.current_bet,
                        player.chips, player.current_bet, self.num_players,
                        sum(1 for p in self.players if not p.folded),
                        self.players.index(player), self.action_history[-10:], 
                        self.opponent_bets[-10:],
                        hand_phase=street_idx
                    )
                raise_amount = player.ai_model.get_raise_size(
                    state, self.pot, self.current_bet, player.chips, 
                    player.current_bet, min_raise
                )
            else:
                # Fallback - use variable sizing based on pot
                max_raise = player.chips - call_amount
                if max_raise > min_raise:
                    # Choose from different strategic sizes
                    options = []
                    if min_raise <= max_raise:
                        options.append(min_raise)
                    if pot_size * 0.33 <= max_raise and pot_size * 0.33 > min_raise:
                        options.append(int(pot_size * 0.33))
                    if pot_size * 0.5 <= max_raise and pot_size * 0.5 > min_raise:
                        options.append(int(pot_size * 0.5))
                    if pot_size * 0.75 <= max_raise and pot_size * 0.75 > min_raise:
                        options.append(int(pot_size * 0.75))
                    if options:
                        raise_amount = random.choice(options)
                    else:
                        raise_amount = min_raise
                else:
                    raise_amount = min_raise
            
            # Execute the raise
            old_bet = self.current_bet
            total = player.bet(call_amount + raise_amount)
            self.pot += total
            self.current_bet = player.current_bet
            
            # Track raise amount for minimum raise rules
            self.last_raise_amount = self.current_bet - old_bet
            
            self.opponent_bets.append(total)
        elif action == Action.ALL_IN:
            amount = player.bet(player.chips)
            self.pot += amount
            if player.current_bet > self.current_bet:
                old_bet = self.current_bet
                self.current_bet = player.current_bet
                self.last_raise_amount = self.current_bet - old_bet
            self.opponent_bets.append(amount)
    
    def determine_winners(self):
        """Determine winners of the hand"""
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            return active
        
        # Fill community cards if needed
        while len(self.community_cards) < 5:
            card = self.deck.draw()
            if card:
                self.community_cards.append(card)
            else:
                break  # Deck is empty
        
        scores = []
        for player in active:
            # Filter out None cards (in case deck ran out)
            all_cards = [c for c in player.hand + self.community_cards if c is not None]
            if len(all_cards) < 5:
                print(f"WARNING: Not enough cards for {player.name} (only {len(all_cards)} cards)")
                score = 0
            else:
                score = evaluate_hand(all_cards)
            scores.append((player, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        max_score = scores[0][1]
        
        return [p for p, s in scores if s == max_score]

# Keep simplified game class for interactive play
class TexasHoldEm:
    def __init__(self, num_players=4, starting_chips=1000, verbose=True):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.verbose = verbose  # Control output during evaluation
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.small_blind = 10
        self.big_blind = 20
        self.dealer_position = 0
        self.players = []
        self.action_history = []
        self.opponent_bets = []
    
    def setup_players(self, ai_model=None):
        """Setup players for interactive game"""
        self.players = []
        
        # Human player
        self.players.append(Player("You", self.starting_chips, is_ai=False))
        
        # AI players
        for i in range(self.num_players - 1):
            ai_player = Player(f"AI_{i+1}", self.starting_chips, is_ai=True)
            if ai_model:
                ai_player.ai_model = ai_model
            self.players.append(ai_player)
    
    def play_hand(self, verbose=True):
        """Play a complete hand interactively"""
        # Debug: Track total chips at start
        initial_total_chips = sum(p.chips for p in self.players)
        if initial_total_chips == 0:
            print(f"CRITICAL WARNING: Starting hand with 0 total chips!")
            print(f"  This should NEVER happen in poker - chips cannot disappear!")
            print(f"  Player chips: {[p.chips for p in self.players]}")
            import traceback
            print("  Stack trace:")
            traceback.print_stack()
            return  # Don't continue with invalid game state

        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.action_history = []
        self.opponent_bets = []
        self.last_raise_amount = self.big_blind  # Initialize minimum raise

        for player in self.players:
            player.reset_hand()
        
        # Deal cards
        for player in self.players:
            if player.chips > 0:
                player.hand = self.deck.draw(2)
        
        # Post blinds
        sb_pos = (self.dealer_position + 1) % len(self.players)
        bb_pos = (self.dealer_position + 2) % len(self.players)
        
        self.pot += self.players[sb_pos].bet(self.small_blind)
        self.pot += self.players[bb_pos].bet(self.big_blind)
        self.current_bet = self.big_blind
        
        # Betting rounds
        streets = ['preflop', 'flop', 'turn', 'river']
        for street_idx, street in enumerate(streets):
            if street == 'flop':
                # Draw 3 cards and filter out None values
                flop_cards = self.deck.draw(3)
                self.community_cards.extend([c for c in flop_cards if c is not None])
            elif street == 'turn':
                card = self.deck.draw()
                if card:
                    self.community_cards.append(card)
            elif street == 'river':
                card = self.deck.draw()
                if card:
                    self.community_cards.append(card)
            
            if self.verbose:
                print(f"\n--- {street.capitalize()} ---")
                if self.community_cards:
                    # Filter out None values for display
                    visible_cards = [c for c in self.community_cards if c is not None]
                    if visible_cards:
                        print(f"Community cards: {visible_cards}")
            
            if not self.betting_round(verbose=verbose, street_idx=street_idx):
                break
            
            if sum(1 for p in self.players if not p.folded) == 1:
                break
        
        # Showdown
        winners = self.determine_winners(verbose=verbose)
        self.distribute_pot(winners, verbose=verbose)

        # Debug: Check chip conservation
        final_total_chips = sum(p.chips for p in self.players) + self.pot
        if final_total_chips != initial_total_chips:
            print(f"\nCHIP CONSERVATION ERROR!")
            print(f"  Initial total: {initial_total_chips}")
            print(f"  Final total: {final_total_chips}")
            print(f"  Difference: {final_total_chips - initial_total_chips}")
            print(f"  Pot remaining: {self.pot}")
            print(f"  Player chips: {[p.chips for p in self.players]}")

        # Check for all players having zero chips
        if all(p.chips == 0 for p in self.players):
            print(f"\nCRITICAL WARNING: All players have 0 chips after hand!")
            print(f"  This should NEVER happen - chips cannot all disappear!")
            print(f"  Initial total was: {initial_total_chips}")
            print(f"  Pot remaining: {self.pot}")
            import traceback
            print("  Stack trace:")
            traceback.print_stack()

        # Move dealer
        self.dealer_position = (self.dealer_position + 1) % len(self.players)
    
    def betting_round(self, verbose=True, street_idx=0):
        """Interactive betting round"""
        first_to_act = (self.dealer_position + 3) % len(self.players) if street_idx == 0 else (self.dealer_position + 1) % len(self.players)
        current = first_to_act
        last_raiser = None  # Track who made the last raise
        
        # Track who has acted since the last raise
        players_to_act = set(p for p in self.players if not p.folded and not p.all_in and p.chips > 0)
        
        max_iterations = len(self.players) * 4  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            player = self.players[current]
            
            # Skip players who can't act
            if player.folded or player.all_in or player.chips == 0:
                current = (current + 1) % len(self.players)
                continue
            
            # Check if this player needs to act
            needs_to_act = False
            if player.current_bet < self.current_bet:
                # Player hasn't matched the current bet
                needs_to_act = True
            elif last_raiser is None and player in players_to_act:
                # No raises yet, player hasn't acted
                needs_to_act = True
            elif last_raiser and player != last_raiser and player in players_to_act:
                # There was a raise and this player hasn't responded to it
                needs_to_act = True
            
            if not needs_to_act:
                current = (current + 1) % len(self.players)
                # Check if we've gone full circle back to the last raiser
                if last_raiser and current == self.players.index(last_raiser):
                    break
                # Check if everyone has matched the bet
                active = [p for p in self.players if not p.folded and not p.all_in and p.chips > 0]
                if all(p.current_bet == self.current_bet for p in active):
                    # Check if we've returned to first to act with no raises
                    if current == first_to_act and last_raiser is None:
                        break
                continue
            
            # Get valid actions
            valid_actions = self.get_valid_actions(player)
            
            # Get action based on player type
            if player.is_ai:
                if player.ai_model:
                    # Check if this is a CFR or Random AI
                    is_cfr_or_random = (hasattr(player.ai_model, '__class__') and 
                                       ('CFRWrapper' in player.ai_model.__class__.__name__ or 
                                        'RandomAI' in player.ai_model.__class__.__name__))
                    
                    if is_cfr_or_random:
                        # CFR and Random AIs use different state format
                        class DummyState:
                            def __init__(self, player, game):
                                self.hole_cards = player.hand
                                self.community_cards = game.community_cards
                                self.pot_size = game.pot
                                self.to_call = max(0, game.current_bet - player.current_bet)
                                self.stack_size = player.chips
                                self.position = game.players.index(player)
                                self.num_players = game.num_players
                                self.action_history = game.action_history[-10:]
                        
                        state = DummyState(player, self)
                    else:
                        # AI with trained model - use state features
                        state = player.ai_model.get_state_features(
                            player.hand, self.community_cards, self.pot, self.current_bet,
                            player.chips, player.current_bet, self.num_players,
                            sum(1 for p in self.players if not p.folded),
                            current, self.action_history[-10:], self.opponent_bets[-10:],
                            hand_phase=street_idx
                        )
                    action = player.ai_model.choose_action(state, valid_actions, training=False)
                    # Store current street for execute_action
                    self.current_street = street_idx
                else:
                    # Random AI (no model)
                    action = random.choice(valid_actions)
            else:
                # Human player
                action = self.get_human_action(player, valid_actions)
            
            # Execute action
            old_bet = self.current_bet
            self.execute_action(player, action, verbose)
            self.action_history.append(action)
            
            # Remove player from players_to_act since they've acted
            players_to_act.discard(player)
            
            # Handle raise/all-in
            if action in [Action.RAISE, Action.ALL_IN] and self.current_bet > old_bet:
                # This was a raise - everyone else needs to act again
                last_raiser = player
                players_to_act = set(p for p in self.players 
                                    if not p.folded and not p.all_in and p.chips > 0 and p != player)
            
            # Check end conditions
            active = [p for p in self.players if not p.folded]
            if len(active) <= 1:
                break
            
            # Check if only all-in players remain (plus maybe one active)
            active_with_chips = [p for p in active if not p.all_in and p.chips > 0]
            if len(active_with_chips) <= 1:
                # If there's one player with chips and others are all-in, they need to match
                if len(active_with_chips) == 1:
                    remaining_player = active_with_chips[0]
                    if remaining_player.current_bet >= self.current_bet:
                        # They've matched, we're done
                        break
                    # Otherwise, continue so they can act
                else:
                    # Everyone is all-in
                    break
            
            # Move to next player
            current = (current + 1) % len(self.players)
            
            # Check if betting is complete
            if not players_to_act:
                # Everyone has acted since the last raise
                active_not_allin = [p for p in self.players if not p.folded and not p.all_in and p.chips > 0]
                if all(p.current_bet == self.current_bet for p in active_not_allin):
                    break
        
        # Reset current bets for next street
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        
        return sum(1 for p in self.players if not p.folded) > 1
    
    def get_valid_actions(self, player):
        """Get valid actions for a player"""
        actions = []
        call_amount = self.current_bet - player.current_bet
        
        if player.chips == 0:
            return [Action.CHECK]
        
        actions.append(Action.FOLD)
        
        if call_amount == 0:
            actions.append(Action.CHECK)
        elif call_amount < player.chips:
            actions.append(Action.CALL)
        
        if player.chips > call_amount * 2:
            actions.append(Action.RAISE)
        
        if player.chips > 0:
            actions.append(Action.ALL_IN)
        
        return actions
    
    def execute_action(self, player, action, verbose):
        """Execute player action with proper no-limit rules"""
        if action == Action.FOLD:
            player.folded = True
            if self.verbose:
                print(f"{player.name} folds")
        elif action == Action.CHECK:
            # Validate that checking is actually allowed
            call_amount = self.current_bet - player.current_bet
            if call_amount > 0:
                # Invalid check - player must call, raise, or fold when there's a bet
                # Convert invalid check to fold
                player.folded = True
                if verbose:
                    print(f"WARNING: {player.name} attempted invalid check with ${call_amount} to call - folding instead")
                return  # Exit early since we converted to fold
            if verbose:
                print(f"{player.name} checks")
        elif action == Action.CALL:
            call_amount = self.current_bet - player.current_bet
            amount = player.bet(call_amount)
            self.pot += amount
            self.opponent_bets.append(amount)
            if self.verbose:
                print(f"{player.name} calls ${amount}")
        elif action == Action.RAISE:
            # Get raise amount
            if not player.is_ai and hasattr(self, 'human_raise_amount'):
                # Use the amount we got from get_human_action
                raise_amount = self.human_raise_amount
                delattr(self, 'human_raise_amount')  # Clean up
            else:
                # AI strategic raise sizing
                call_amount = self.current_bet - player.current_bet
                pot_size = self.pot + call_amount
                
                # Determine minimum raise
                if hasattr(self, 'last_raise_amount'):
                    min_raise = self.last_raise_amount
                else:
                    min_raise = self.big_blind
                
                # Get strategic raise size from AI
                if player.ai_model and hasattr(player.ai_model, 'get_raise_size'):
                    # Check if this is a CFR or Random AI
                    is_cfr_or_random = (hasattr(player.ai_model, '__class__') and 
                                       ('CFRWrapper' in player.ai_model.__class__.__name__ or 
                                        'RandomAI' in player.ai_model.__class__.__name__))
                    
                    if is_cfr_or_random:
                        # CFR and Random AIs use different state format
                        class DummyState:
                            def __init__(self, player, game):
                                self.hole_cards = player.hand
                                self.community_cards = game.community_cards
                                self.pot_size = game.pot
                                self.to_call = max(0, game.current_bet - player.current_bet)
                                self.stack_size = player.chips
                                self.position = game.players.index(player)
                                self.num_players = game.num_players
                                self.action_history = game.action_history[-10:]
                        
                        state = DummyState(player, self)
                    else:
                        # Regular AI - get current state for the AI
                        hand_phase = self.current_street if hasattr(self, 'current_street') else 0
                        state = player.ai_model.get_state_features(
                            player.hand, self.community_cards, self.pot, self.current_bet,
                            player.chips, player.current_bet, self.num_players,
                            sum(1 for p in self.players if not p.folded),
                            self.players.index(player), self.action_history[-10:], 
                            self.opponent_bets[-10:],
                            hand_phase=hand_phase
                        )
                    raise_amount = player.ai_model.get_raise_size(
                        state, self.pot, self.current_bet, player.chips, 
                        player.current_bet, min_raise
                    )
                else:
                    # Fallback for untrained AI - simple strategy
                    max_raise = player.chips - call_amount
                    # Random AI chooses between different sizes
                    if max_raise > min_raise:
                        options = [min_raise]
                        if pot_size * 0.5 <= max_raise:
                            options.append(int(pot_size * 0.5))
                        if pot_size * 0.75 <= max_raise:
                            options.append(int(pot_size * 0.75))
                        raise_amount = random.choice(options)
                    else:
                        raise_amount = min_raise
            
            # Execute the raise
            call_amount = self.current_bet - player.current_bet
            old_bet = self.current_bet
            total = player.bet(call_amount + raise_amount)
            self.pot += total
            self.current_bet = player.current_bet
            
            # Track raise amount for minimum raise rules
            self.last_raise_amount = self.current_bet - old_bet
            
            self.opponent_bets.append(total)
            if self.verbose:
                if old_bet > 0:
                    print(f"{player.name} raises to ${self.current_bet}")
                else:
                    print(f"{player.name} bets ${self.current_bet}")
                    
        elif action == Action.ALL_IN:
            amount = player.bet(player.chips)
            self.pot += amount
            old_bet = self.current_bet
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
                # Track raise amount if this all-in is effectively a raise
                self.last_raise_amount = self.current_bet - old_bet
                if self.verbose:
                    if old_bet > 0:
                        print(f"{player.name} goes all-in for ${amount} (raises to ${self.current_bet})")
                    else:
                        print(f"{player.name} goes all-in for ${amount}")
            else:
                if self.verbose:
                    print(f"{player.name} goes all-in for ${amount} (call)")
            self.opponent_bets.append(amount)
    
    def get_human_action(self, player, valid_actions):
        """Get action from human player"""
        print(f"\n{player.name}'s turn")
        print(f"Your cards: {player.hand}")
        print(f"Pot: ${self.pot}, Your chips: ${player.chips}")
        print(f"Current bet: ${self.current_bet}, Your bet: ${player.current_bet}")
        
        action_map = {str(i): action for i, action in enumerate(valid_actions)}
        for i, action in action_map.items():
            if action == Action.CALL:
                print(f"{i}: CALL ${self.current_bet - player.current_bet}")
            elif action == Action.RAISE:
                call_amount = self.current_bet - player.current_bet
                min_raise = self.last_raise_amount if hasattr(self, 'last_raise_amount') else self.big_blind
                min_total = call_amount + min_raise
                print(f"{i}: RAISE (minimum ${min_raise} on top of ${call_amount} call)")
            else:
                print(f"{i}: {action.name}")
        
        while True:
            choice = input("Choose action: ")
            if choice in action_map:
                action = action_map[choice]
                
                # If RAISE is chosen, prompt for amount
                if action == Action.RAISE:
                    call_amount = self.current_bet - player.current_bet
                    min_raise = self.last_raise_amount if hasattr(self, 'last_raise_amount') else self.big_blind
                    max_raise = player.chips - call_amount
                    pot_size = self.pot + call_amount
                    
                    print(f"\nRaise sizing options:")
                    print(f"  Current pot (including call): ${pot_size}")
                    print(f"  Minimum raise: ${min_raise}")
                    print(f"  Maximum raise: ${max_raise} (all-in)")
                    
                    # Show common raise sizes
                    sizes = []
                    sizes.append(("Minimum", min_raise))
                    
                    one_third = int(pot_size * 0.33)
                    if one_third > min_raise:
                        sizes.append(("1/3 pot", one_third))
                    
                    half_pot = int(pot_size * 0.5)
                    if half_pot > min_raise and half_pot <= max_raise:
                        sizes.append(("1/2 pot", half_pot))
                    
                    three_quarters = int(pot_size * 0.75)
                    if three_quarters > min_raise and three_quarters <= max_raise:
                        sizes.append(("3/4 pot", three_quarters))
                    
                    full_pot = pot_size
                    if full_pot > min_raise and full_pot <= max_raise:
                        sizes.append(("Pot-sized", full_pot))
                    
                    overbet = int(pot_size * 1.5)
                    if overbet > min_raise and overbet <= max_raise:
                        sizes.append(("1.5x pot", overbet))
                    
                    print("\nSuggested sizes:")
                    for i, (name, size) in enumerate(sizes):
                        print(f"  {i}: {name} = ${size}")
                    print(f"  Or enter custom amount (${min_raise}-${max_raise})")
                    
                    while True:
                        raise_input = input("Choose size or enter amount: ")
                        
                        # Check if it's a suggested size index
                        if raise_input.isdigit() and int(raise_input) < len(sizes):
                            raise_amount = sizes[int(raise_input)][1]
                            break
                        
                        # Try to parse as custom amount
                        try:
                            raise_amount = int(raise_input)
                            if min_raise <= raise_amount <= max_raise:
                                break
                            else:
                                print(f"Invalid amount. Must be between ${min_raise} and ${max_raise}")
                        except ValueError:
                            print("Invalid input. Enter a number.")
                    
                    # Store the raise amount for execute_action to use
                    self.human_raise_amount = raise_amount
                
                return action
            print("Invalid choice.")
    
    def determine_winners(self, verbose=True):
        """Determine winners"""
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            return active
        
        # Fill community cards
        while len(self.community_cards) < 5:
            card = self.deck.draw()
            if card:
                self.community_cards.append(card)
            else:
                break  # Deck is empty
        
        scores = []
        for player in active:
            # Filter out None cards (in case deck ran out)
            all_cards = [c for c in player.hand + self.community_cards if c is not None]
            if len(all_cards) < 5:
                print(f"WARNING: Not enough cards for {player.name} (only {len(all_cards)} cards)")
                score = 0
            else:
                score = evaluate_hand(all_cards)
            scores.append((player, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        max_score = scores[0][1]
        
        if verbose:
            print("\n--- Showdown ---")
            for player, score in scores:
                print(f"{player.name}: {player.hand} - Score: {score}")
        
        return [p for p, s in scores if s == max_score]
    
    def distribute_pot(self, winners, verbose=True):
        """Distribute pot to winners"""
        if not winners:
            return

        split = self.pot // len(winners)
        remainder = self.pot % len(winners)

        if verbose:
            print(f"\n--- Pot Distribution ---")
            print(f"Total pot: ${self.pot}")

        for i, winner in enumerate(winners):
            amount = split + (1 if i < remainder else 0)
            winner.chips += amount
            if verbose:
                print(f"{winner.name} wins ${amount}")

        # CRITICAL FIX: Reset pot to 0 after distribution
        self.pot = 0