"""
CPU版遗传算法交易员训练主程序
兼容Python 3.13，无需GPU依赖
"""

import argparse
import logging
import time
from pathlib import Path
import json
import numpy as np
import sys
import os
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 确保results目录存在
results_dir = Path('../results')
results_dir.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(results_dir / 'training_cpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CPUGAConfig:
    """CPU版遗传算法配置"""
    population_size: int = 200
    gene_length: int = 1405
    feature_dim: int = 1400
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elite_ratio: float = 0.1
    tournament_size: int = 5
    max_generations: int = 200
    early_stop_patience: int = 20
    batch_size: int = 50
    n_jobs: int = -1

class CPUDataProcessor:
    """CPU版数据处理器"""
    
    def __init__(self, window_size: int = 350, normalization: str = 'relative'):
        self.window_size = window_size
        self.normalization = normalization
        
    def load_and_process_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载和处理数据"""
        import pandas as pd
        
        logger.info(f"加载数据文件: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"数据形状: {df.shape}")
            
            # 检查必要的列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")
            
            # 提取OHLCV数据
            ohlcv = df[required_cols].values.astype(np.float32)
            
            # 归一化处理
            features = self._normalize_data(ohlcv)
            
            # 创建滑动窗口
            windowed_features = self._create_windows(features)
            
            # 创建标签（下一期收益率）
            returns = self._calculate_returns(df['Close'].values)
            labels = returns[self.window_size:]
            
            logger.info(f"处理后特征形状: {windowed_features.shape}")
            logger.info(f"标签形状: {labels.shape}")
            
            return windowed_features, labels
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            raise
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据归一化"""
        if self.normalization == 'relative':
            normalized = np.zeros_like(data)
            normalized[:, :-1] = data[:, :-1] / data[:, [3]]
            normalized[:, -1] = data[:, -1] / np.mean(data[:, -1])
            return normalized
        elif self.normalization == 'rolling':
            window = min(20, len(data) // 10)
            normalized = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i+1]
                price_mean = np.mean(window_data[:, :4])
                price_std = np.std(window_data[:, :4]) + 1e-8
                normalized[i, :4] = (data[i, :4] - price_mean) / price_std
                vol_mean = np.mean(window_data[:, 4])
                vol_std = np.std(window_data[:, 4]) + 1e-8
                normalized[i, 4] = (data[i, 4] - vol_mean) / vol_std
            return normalized
        else:
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            return (data - data_min) / (data_max - data_min + 1e-8)
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """创建滑动窗口"""
        n_samples = len(data) - self.window_size + 1
        windows = np.zeros((n_samples, self.window_size * data.shape[1]))
        for i in range(n_samples):
            window = data[i:i + self.window_size]
            windows[i] = window.flatten()
        return windows
    
    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """计算收益率"""
        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        return returns

class CPUGeneticAlgorithm:
    """CPU版遗传算法"""
    
    def __init__(self, config: CPUGAConfig):
        self.config = config
        self.population = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.random.randn(self.config.population_size, self.config.gene_length)
        population[:, :self.config.feature_dim] *= 0.1
        population[:, self.config.feature_dim:] = np.random.uniform(
            [0.01, 0.02, 0.1, 0.1, 0.1],
            [0.1, 0.2, 1.0, 2.0, 1.0],
            (self.config.population_size, 5)
        )
        self.population = population
        return population
    
    def evaluate_fitness(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        fitness_scores = np.zeros(self.config.population_size)
        for i in range(self.config.population_size):
            individual = self.population[i]
            fitness_scores[i] = self._evaluate_individual(individual, features, labels)
        return fitness_scores
    
    def _evaluate_individual(self, individual: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        """评估单个个体"""
        try:
            feature_weights = individual[:self.config.feature_dim]
            risk_params = individual[self.config.feature_dim:]
            signals = np.dot(features, feature_weights)
            returns = self._simulate_trading(signals, labels, risk_params)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                return sharpe_ratio
            else:
                return -10.0
        except Exception as e:
            logger.warning(f"个体评估失败: {e}")
            return -10.0
    
    def _simulate_trading(self, signals: np.ndarray, actual_returns: np.ndarray, risk_params: np.ndarray) -> List[float]:
        """简化的交易模拟"""
        stop_loss, take_profit, max_position, risk_factor, trade_freq = risk_params
        portfolio_returns = []
        position = 0.0
        
        for i in range(len(signals)):
            signal_strength = np.tanh(signals[i])
            target_position = signal_strength * max_position
            position_change = target_position - position
            
            if abs(position_change) < trade_freq:
                position_change = 0
            
            position += position_change
            
            if i < len(actual_returns):
                period_return = position * actual_returns[i]
                if period_return < -stop_loss:
                    position = 0
                    period_return = -stop_loss
                elif period_return > take_profit:
                    position *= 0.5
                portfolio_returns.append(period_return)
        
        return portfolio_returns
    
    def selection(self, fitness_scores: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        selected = np.zeros_like(self.population)
        for i in range(self.config.population_size):
            tournament_indices = np.random.choice(
                self.config.population_size, 
                self.config.tournament_size, 
                replace=False
            )
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected[i] = self.population[winner_idx].copy()
        return selected
    
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """交叉操作"""
        offspring = parents.copy()
        for i in range(0, self.config.population_size - 1, 2):
            if np.random.random() < self.config.crossover_rate:
                crossover_point = np.random.randint(1, self.config.gene_length)
                offspring[i, crossover_point:] = parents[i+1, crossover_point:]
                offspring[i+1, crossover_point:] = parents[i, crossover_point:]
        return offspring
    
    def mutation(self, individuals: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individuals.copy()
        for i in range(self.config.population_size):
            for j in range(self.config.gene_length):
                if np.random.random() < self.config.mutation_rate:
                    if j < self.config.feature_dim:
                        mutated[i, j] += np.random.normal(0, 0.01)
                    else:
                        param_idx = j - self.config.feature_dim
                        if param_idx == 0:
                            mutated[i, j] = np.clip(mutated[i, j] + np.random.normal(0, 0.005), 0.01, 0.1)
                        elif param_idx == 1:
                            mutated[i, j] = np.clip(mutated[i, j] + np.random.normal(0, 0.01), 0.02, 0.2)
                        elif param_idx == 2:
                            mutated[i, j] = np.clip(mutated[i, j] + np.random.normal(0, 0.05), 0.1, 1.0)
                        elif param_idx == 3:
                            mutated[i, j] = np.clip(mutated[i, j] + np.random.normal(0, 0.1), 0.1, 2.0)
                        else:
                            mutated[i, j] = np.clip(mutated[i, j] + np.random.normal(0, 0.05), 0.1, 1.0)
        return mutated
    
    def evolve(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """进化过程"""
        logger.info("开始CPU版遗传算法进化")
        self.initialize_population()
        start_time = time.time()
        
        for generation in range(self.config.max_generations):
            gen_start_time = time.time()
            fitness_scores = self.evaluate_fitness(features, labels)
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            })
            
            selected = self.selection(fitness_scores)
            elite_count = int(self.config.population_size * self.config.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            offspring = self.crossover(selected)
            offspring = self.mutation(offspring)
            
            for i, elite_idx in enumerate(elite_indices):
                offspring[i] = self.population[elite_idx].copy()
            
            self.population = offspring
            gen_time = time.time() - gen_start_time
            
            if generation % 10 == 0:
                logger.info(f"代数 {generation}: 最佳适应度={self.best_fitness:.4f}, "
                          f"平均适应度={np.mean(fitness_scores):.4f}, "
                          f"用时={gen_time:.2f}s")
            
            if generation > self.config.early_stop_patience:
                recent_best = [h['best_fitness'] for h in self.fitness_history[-self.config.early_stop_patience:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    logger.info(f"早停于第{generation}代")
                    break
        
        total_time = time.time() - start_time
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'total_time': total_time,
            'final_generation': generation
        }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CPU版遗传算法交易员训练')
    parser.add_argument('--data_file', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--window_size', type=int, default=350, help='滑动窗口大小')
    parser.add_argument('--normalization', type=str, default='relative', choices=['relative', 'rolling', 'minmax'], help='归一化方法')
    parser.add_argument('--population_size', type=int, default=200, help='种群大小')
    parser.add_argument('--generations', type=int, default=200, help='最大进化代数')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='变异率')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='交叉率')
    parser.add_argument('--output_dir', type=str, default='../results', help='结果输出目录')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== CPU版遗传算法交易员训练开始 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"数据文件: {args.data_file}")
    
    if not os.path.exists(args.data_file):
        logger.error(f"数据文件不存在: {args.data_file}")
        return
    
    try:
        processor = CPUDataProcessor(window_size=args.window_size, normalization=args.normalization)
        features, labels = processor.load_and_process_data(args.data_file)
        
        config = CPUGAConfig(
            population_size=args.population_size,
            max_generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            feature_dim=features.shape[1]
        )
        
        ga = CPUGeneticAlgorithm(config)
        results = ga.evolve(features, labels)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        np.save(output_dir / f"best_individual_{timestamp}.npy", results['best_individual'])
        
        with open(output_dir / f"training_history_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(results['fitness_history'], f, indent=2, ensure_ascii=False)
        
        config_dict = {
            'data_file': args.data_file,
            'window_size': args.window_size,
            'normalization': args.normalization,
            'population_size': config.population_size,
            'max_generations': config.max_generations,
            'mutation_rate': config.mutation_rate,
            'crossover_rate': config.crossover_rate,
            'feature_dim': config.feature_dim
        }
        with open(output_dir / f"config_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info("=== 训练完成 ===")
        logger.info(f"最佳适应度: {results['best_fitness']:.4f}")
        logger.info(f"总训练时间: {results['total_time']:.2f}秒")
        
        print("\n" + "="*50)
        print("CPU版遗传算法训练完成")
        print("="*50)
        print(f"最佳夏普率: {results['best_fitness']:.4f}")
        print(f"训练时间: {results['total_time']:.1f}秒")
        print(f"进化代数: {results['final_generation']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()