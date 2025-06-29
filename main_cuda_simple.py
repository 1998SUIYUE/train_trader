#!/usr/bin/env python3
"""
Simplified CUDA Training Script
Avoids complex imports and focuses on core functionality
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_environment():
    """Check if the environment is ready"""
    print("=== Environment Check ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: True")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
            
            return True
        else:
            print("CUDA available: False")
            print("Will use CPU for computation")
            return False
            
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False

def create_demo_data():
    """Create demo trading data"""
    print("\n=== Creating Demo Data ===")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic OHLC data
    np.random.seed(42)
    n_days = 2000
    base_price = 2000.0
    
    # Simulate price random walk
    daily_returns = np.random.normal(0.0005, 0.02, n_days)
    close_prices = base_price * np.exp(np.cumsum(daily_returns))
    
    # Generate OHLC
    opens = np.roll(close_prices, 1)
    opens[0] = base_price
    
    highs = np.maximum(opens, close_prices) * (1 + np.random.exponential(0.005, n_days))
    lows = np.minimum(opens, close_prices) * (1 - np.random.exponential(0.005, n_days))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices
    })
    
    # Save data
    data_file = data_dir / 'demo_data.csv'
    data.to_csv(data_file, index=False)
    
    print(f"Demo data created: {data_file}")
    print(f"Data shape: {data.shape}")
    print(f"Price range: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    return data_file

def simple_feature_extraction(data_file, window_size=100):
    """Simple feature extraction without complex GPU managers"""
    print(f"\n=== Feature Extraction (window_size={window_size}) ===")
    
    import torch
    
    # Load data
    df = pd.read_csv(data_file)
    ohlc_data = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    print(f"Loaded data shape: {ohlc_data.shape}")
    
    # Create sliding windows
    n_samples = len(ohlc_data)
    n_windows = n_samples - window_size + 1
    
    if n_windows <= 0:
        raise ValueError(f"Data length {n_samples} is smaller than window size {window_size}")
    
    print(f"Creating {n_windows} windows...")
    
    # Extract features (simple relative price normalization)
    features = []
    labels = []
    
    for i in range(n_windows):
        window = ohlc_data[i:i+window_size]
        
        # Normalize by first close price
        base_price = window[0, 3]  # First close price
        normalized_window = window / base_price
        
        # Flatten window as features
        feature_vector = normalized_window.flatten()
        features.append(feature_vector)
        
        # Label is next period return (if available)
        if i + window_size < n_samples:
            current_price = ohlc_data[i + window_size - 1, 3]
            next_price = ohlc_data[i + window_size, 3]
            label = (next_price - current_price) / current_price
        else:
            label = 0.0
        
        labels.append(label)
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    
    print(f"Features shape: {features_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Device: {device}")
    
    return features_tensor, labels_tensor

def simple_genetic_algorithm(features, labels, config):
    """Simple genetic algorithm implementation"""
    print(f"\n=== Simple Genetic Algorithm ===")
    
    import torch
    
    device = features.device
    population_size = config['population_size']
    feature_dim = features.shape[1]
    max_generations = config['max_generations']
    
    print(f"Population size: {population_size}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Max generations: {max_generations}")
    print(f"Device: {device}")
    
    # Initialize population (weights + bias + trading thresholds)
    individual_size = feature_dim + 3  # weights + bias + buy_threshold + sell_threshold
    population = torch.randn(population_size, individual_size, device=device) * 0.1
    
    # Initialize trading thresholds
    population[:, feature_dim + 1] = torch.sigmoid(torch.randn(population_size, device=device)) * 0.3 + 0.5  # buy threshold [0.5, 0.8]
    population[:, feature_dim + 2] = torch.sigmoid(torch.randn(population_size, device=device)) * 0.3 + 0.2  # sell threshold [0.2, 0.5]
    
    best_fitness = -float('inf')
    best_individual = None
    
    print("\nStarting evolution...")
    
    for generation in range(max_generations):
        start_time = time.time()
        
        # Evaluate fitness
        fitness_scores = evaluate_population(population, features, labels)
        
        # Track best individual
        best_idx = torch.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx].item()
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[best_idx].clone()
        
        # Selection (tournament)
        new_population = tournament_selection(population, fitness_scores)
        
        # Crossover and mutation
        new_population = crossover_and_mutation(new_population, config)
        
        population = new_population
        
        generation_time = time.time() - start_time
        
        if generation % 10 == 0:
            avg_fitness = torch.mean(fitness_scores).item()
            print(f"Gen {generation:3d} | Best: {current_best_fitness:8.4f} | Avg: {avg_fitness:8.4f} | Time: {generation_time:.3f}s")
    
    print(f"\nEvolution completed!")
    print(f"Best fitness: {best_fitness:.6f}")
    
    return {
        'best_fitness': best_fitness,
        'best_individual': best_individual.cpu().numpy(),
        'final_generation': max_generations
    }

def evaluate_population(population, features, labels):
    """Evaluate population fitness using simple trading simulation"""
    import torch
    
    population_size, individual_size = population.shape
    feature_dim = features.shape[1]
    
    # Extract parameters
    weights = population[:, :feature_dim]  # [pop_size, feature_dim]
    biases = population[:, feature_dim]    # [pop_size]
    buy_thresholds = population[:, feature_dim + 1]   # [pop_size]
    sell_thresholds = population[:, feature_dim + 2]  # [pop_size]
    
    # Compute trading signals
    signals = torch.sigmoid(torch.matmul(weights, features.T) + biases.unsqueeze(1))  # [pop_size, n_samples]
    
    # Simple trading simulation
    fitness_scores = torch.zeros(population_size, device=population.device)
    
    for i in range(population_size):
        individual_signals = signals[i]
        buy_threshold = buy_thresholds[i]
        sell_threshold = sell_thresholds[i]
        
        # Simple trading logic
        position = 0.0
        portfolio_value = 1.0
        returns = []
        
        for t in range(len(individual_signals)):
            signal = individual_signals[t].item()
            price_return = labels[t].item()
            
            # Trading decisions
            if signal > buy_threshold and position == 0:
                position = 1.0  # Buy
            elif signal < sell_threshold and position > 0:
                position = 0.0  # Sell
            
            # Update portfolio
            if position > 0:
                portfolio_return = price_return
            else:
                portfolio_return = 0.0
            
            portfolio_value *= (1 + portfolio_return)
            returns.append(portfolio_return)
        
        # Calculate fitness (simple Sharpe ratio)
        if len(returns) > 0:
            returns_tensor = torch.tensor(returns, device=population.device)
            mean_return = torch.mean(returns_tensor)
            std_return = torch.std(returns_tensor) + 1e-8
            sharpe_ratio = mean_return / std_return
            fitness_scores[i] = sharpe_ratio
    
    return fitness_scores

def tournament_selection(population, fitness_scores, tournament_size=3):
    """Tournament selection"""
    import torch
    
    population_size = population.shape[0]
    new_population = torch.zeros_like(population)
    
    for i in range(population_size):
        # Random tournament
        tournament_indices = torch.randint(0, population_size, (tournament_size,), device=population.device)
        tournament_fitness = fitness_scores[tournament_indices]
        
        # Select winner
        winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
        new_population[i] = population[winner_idx]
    
    return new_population

def crossover_and_mutation(population, config):
    """Crossover and mutation operations"""
    import torch
    
    population_size, individual_size = population.shape
    crossover_rate = config['crossover_rate']
    mutation_rate = config['mutation_rate']
    
    # Crossover
    for i in range(0, population_size - 1, 2):
        if torch.rand(1, device=population.device) < crossover_rate:
            # Uniform crossover
            mask = torch.rand(individual_size, device=population.device) < 0.5
            
            temp = population[i].clone()
            population[i] = torch.where(mask, population[i + 1], population[i])
            population[i + 1] = torch.where(mask, temp, population[i + 1])
    
    # Mutation
    mutation_mask = torch.rand_like(population) < mutation_rate
    mutation_noise = torch.randn_like(population) * 0.01
    population += mutation_mask * mutation_noise
    
    # Ensure trading thresholds stay in valid ranges
    feature_dim = individual_size - 3
    population[:, feature_dim + 1] = torch.clamp(population[:, feature_dim + 1], 0.5, 0.8)  # buy threshold
    population[:, feature_dim + 2] = torch.clamp(population[:, feature_dim + 2], 0.2, 0.5)  # sell threshold
    
    return population

def main():
    """Main function"""
    print("Simplified CUDA Trading Agent Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'population_size': 100,
        'max_generations': 50,
        'crossover_rate': 0.8,
        'mutation_rate': 0.02,
        'window_size': 50,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check environment
    cuda_available = check_environment()
    
    if cuda_available:
        print("\nCUDA is available - using GPU acceleration")
    else:
        print("\nCUDA not available - using CPU")
        # Reduce problem size for CPU
        config['population_size'] = 50
        config['max_generations'] = 20
        config['window_size'] = 30
        print("Reduced configuration for CPU:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    try:
        # Create demo data
        data_file = create_demo_data()
        
        # Extract features
        features, labels = simple_feature_extraction(data_file, config['window_size'])
        
        # Run genetic algorithm
        start_time = time.time()
        results = simple_genetic_algorithm(features, labels, config)
        total_time = time.time() - start_time
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f'simple_training_results_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        save_results = {
            'config': config,
            'best_fitness': float(results['best_fitness']),
            'final_generation': int(results['final_generation']),
            'total_time': float(total_time),
            'cuda_available': cuda_available
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save best individual
        best_individual_file = results_dir / f'best_individual_{timestamp}.npy'
        np.save(best_individual_file, results['best_individual'])
        
        # Final report
        print("\n" + "=" * 50)
        print("Training Completed Successfully!")
        print("=" * 50)
        print(f"Best fitness: {results['best_fitness']:.6f}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per generation: {total_time/config['max_generations']:.3f} seconds")
        print(f"Results saved: {results_file}")
        print(f"Best individual saved: {best_individual_file}")
        print("=" * 50)
        
        if cuda_available:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"Final GPU memory usage: {allocated:.2f} GB")
        
        print("\nNext steps:")
        print("1. Analyze the results in the results/ directory")
        print("2. Try different configurations")
        print("3. Use real trading data instead of demo data")
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)