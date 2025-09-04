import random
from game_constants import Action
from poker_ai import PokerAI
from game_engine import TexasHoldEm
from training import hyperparameter_tuning, train_ai_advanced, evaluate_ai_full

def main():
    """Main game loop"""
    print("=" * 60)
    print("TEXAS HOLD'EM with Advanced PyTorch AI")
    print("=" * 60)
    
    print("\n1. Train new AI (standard)")
    print("2. Train with hyperparameter tuning (recommended)")
    print("3. Load existing AI")
    print("4. Play without AI")
    
    choice = input("\nChoose option (1-4): ")
    
    ai_model = None
    
    if choice == '1':
        episodes = int(input("Training episodes (1000-3000 recommended): ") or "1000")
        players = int(input("Number of players (2-6): ") or "4")
        
        ai_model = train_ai_advanced(num_episodes=episodes)
        
        # Evaluate
        eval_choice = input("\nEvaluate AI? (y/n): ")
        if eval_choice.lower() == 'y':
            evaluate_ai_full(ai_model, num_games=100, num_players=players)
        
        # Save
        save_choice = input("\nSave model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: poker_ai.pth): ") or "poker_ai.pth"
            ai_model.save(filename)
            print(f"Model saved to {filename}")
    
    elif choice == '2':
        print("\nStarting hyperparameter tuning...")
        print("This will test multiple configurations to find the best one.")
        
        num_configs = int(input("Number of configurations to test (3-10): ") or "5")
        episodes = int(input("Training episodes per config (100-500): ") or "200")
        
        best_config, results = hyperparameter_tuning(
            num_configs=num_configs,
            episodes_per_config=episodes,
            eval_games=30
        )
        
        # Train final model with best config
        print("\nTraining final model with best configuration...")
        final_episodes = int(input("Final training episodes (1000-5000): ") or "2000")
        
        ai_model = train_ai_advanced(
            num_episodes=final_episodes,
            config=best_config
        )
        
        # Full evaluation
        print("\nPerforming full evaluation...")
        evaluate_ai_full(ai_model, num_games=100, num_players=4)
        
        # Save
        save_choice = input("\nSave model? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Filename (default: poker_ai_tuned.pth): ") or "poker_ai_tuned.pth"
            ai_model.save(filename)
            print(f"Model saved to {filename}")
    
    elif choice == '3':
        filename = input("Model filename (default: poker_ai.pth): ") or "poker_ai.pth"
        ai_model = PokerAI()
        try:
            ai_model.load(filename)
            print(f"Model loaded from {filename}")
            ai_model.epsilon = 0  # No exploration during play
            
            # Option to evaluate
            eval_choice = input("\nEvaluate loaded model? (y/n): ")
            if eval_choice.lower() == 'y':
                players = int(input("Number of players for evaluation (2-6): ") or "4")
                evaluate_ai_full(ai_model, num_games=100, num_players=players)
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with untrained AI")
    
    # Play game
    play_choice = input("\nPlay interactive game? (y/n): ")
    if play_choice.lower() != 'y':
        print("Thanks for using the poker AI trainer!")
        return
    
    print("\n--- Game Setup ---")
    num_players = int(input("Number of players (2-6): ") or "4")
    starting_chips = int(input("Starting chips (default 1000): ") or "1000")
    
    game = TexasHoldEm(num_players=num_players, starting_chips=starting_chips)
    
    if ai_model:
        ai_model.epsilon = 0  # No exploration during play
        game.setup_players(ai_model)
    else:
        game.setup_players()
    
    # Game loop
    print("\n" + "=" * 60)
    print("STARTING GAME")
    print("=" * 60)
    
    hand_count = 0
    while True:
        # Check game over
        active = [p for p in game.players if p.chips > 0]
        if len(active) <= 1:
            winner = active[0] if active else None
            if winner:
                print(f"\n🏆 GAME OVER! {winner.name} wins!")
            else:
                print("\n🏆 GAME OVER! Draw!")
            
            # Final standings
            print("\nFinal standings:")
            sorted_players = sorted(game.players, key=lambda p: p.chips, reverse=True)
            for i, player in enumerate(sorted_players, 1):
                print(f"  {i}. {player.name}: ${player.chips}")
            break
        
        hand_count += 1
        print(f"\n{'='*60}")
        print(f"HAND #{hand_count}")
        print(f"{'='*60}")
        
        # Show chip counts
        print("\nChip counts:")
        for player in game.players:
            if player.chips > 0:
                status = " (Dealer)" if game.players.index(player) == game.dealer_position else ""
                print(f"  {player.name}: ${player.chips}{status}")
        
        # Play hand
        game.play_hand()
        
        # Check if human eliminated
        if game.players[0].chips <= 0:
            print("\n❌ You're out of chips!")
            print("\nFinal standings:")
            sorted_players = sorted(game.players, key=lambda p: p.chips, reverse=True)
            for i, player in enumerate(sorted_players, 1):
                print(f"  {i}. {player.name}: ${player.chips}")
            break
        
        # Continue?
        cont = input("\nPress Enter to continue (or 'q' to quit): ")
        if cont.lower() == 'q':
            break
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()