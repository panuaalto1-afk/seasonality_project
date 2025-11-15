# backtest_scripts/optimizer.py
"""
Walk-Forward Optimizer for Backtesting
Optimizes strategy parameters using Bayesian optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import date, timedelta
from sklearn.model_selection import TimeSeriesSplit
import json
import os

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("[WARN] scikit-optimize not installed. Install with: pip install scikit-optimize")

class WalkForwardOptimizer:
    """
    Walk-forward optimization for backtest parameters
    Uses Bayesian optimization to find best parameters
    """
    
    def __init__(self, 
                 backtest_engine_factory: Callable,
                 param_space: Dict,
                 optimization_config: Dict):
        """
        Initialize optimizer
        
        Args:
            backtest_engine_factory: Function that creates BacktestEngine instance
            param_space: Dict defining parameter search space
            optimization_config: Optimization settings
        """
        self.engine_factory = backtest_engine_factory
        self.param_space = param_space
        self.config = optimization_config
        
        print(f"[Optimizer] Initialized")
        print(f"  Method: {optimization_config.get('method', 'bayesian')}")
        print(f"  Max iterations: {optimization_config.get('max_iterations', 100)}")
    
    def optimize(self, 
                 start_date: date, 
                 end_date: date,
                 objective: str = 'sharpe') -> Dict:
        """
        Run walk-forward optimization
        
        Args:
            start_date: Optimization period start
            end_date: Optimization period end
            objective: Metric to optimize ('sharpe', 'return', 'calmar')
        
        Returns:
            dict with best parameters and results
        """
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Objective: Maximize {objective}")
        print("")
        
        method = self.config.get('method', 'grid')
        
        if method == 'bayesian' and SKOPT_AVAILABLE:
            return self._optimize_bayesian(start_date, end_date, objective)
        elif method == 'grid':
            return self._optimize_grid(start_date, end_date, objective)
        elif method == 'random':
            return self._optimize_random(start_date, end_date, objective)
        else:
            print(f"[WARN] Method '{method}' not available, using grid search")
            return self._optimize_grid(start_date, end_date, objective)
    
    def _optimize_bayesian(self, start_date: date, end_date: date, objective: str) -> Dict:
        """Bayesian optimization using scikit-optimize"""
        print("[Optimizer] Using Bayesian optimization (skopt)")
        
        # Define search space for skopt
        space = []
        param_names = []
        
        for param_name, param_def in self.param_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous range
                space.append(Real(param_def[0], param_def[1], name=param_name))
                param_names.append(param_name)
            elif isinstance(param_def, list):
                # Categorical
                if all(isinstance(x, (int, float)) for x in param_def):
                    space.append(Integer(min(param_def), max(param_def), name=param_name))
                else:
                    space.append(Categorical(param_def, name=param_name))
                param_names.append(param_name)
        
        # Objective function
        iteration = [0]
        best_score = [-np.inf]
        
        @use_named_args(space)
        def objective_function(**params):
            iteration[0] += 1
            
            # Run backtest with these parameters
            config_overrides = {
                'start_date': start_date,
                'end_date': end_date,
            }
            config_overrides.update(params)
            
            try:
                engine = self.engine_factory(config_overrides=config_overrides)
                results = engine.run()
                
                # Extract objective metric
                analysis = results['analysis']
                pm = analysis.get('portfolio_metrics', {})
                
                if objective == 'sharpe':
                    score = pm.get('sharpe_ratio', -999.0)
                    if score == 0.0:  # Invalid
                        score = -999.0
                elif objective == 'return':
                    score = pm.get('annual_return_pct', 0.0) / 100
                elif objective == 'calmar':
                    score = pm.get('calmar_ratio', 0.0)
                else:
                    score = pm.get('sharpe_ratio', 0.0)
                
                # Track best
                if score > best_score[0]:
                    best_score[0] = score
                
                print(f"  Iteration {iteration[0]}: {objective} = {score:.4f} (best: {best_score[0]:.4f})")
                
                # Return negative score (skopt minimizes)
                return -score
            
            except Exception as e:
                print(f"  Iteration {iteration[0]}: FAILED - {e}")
                return 1e10  # Large penalty
        
        # Run optimization
        max_iter = self.config.get('max_iterations', 50)
        n_jobs = self.config.get('n_jobs', 1)
        
        result = gp_minimize(
            objective_function,
            space,
            n_calls=max_iter,
            n_jobs=n_jobs,
            random_state=42,
            verbose=False
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_objective_value = -result.fun
        
        print(f"\n[Optimizer] Optimization complete!")
        print(f"  Best {objective}: {best_objective_value:.4f}")
        print(f"  Best parameters: {json.dumps(best_params, indent=2)}")
        
        return {
            'best_params': best_params,
            'best_score': best_objective_value,
            'objective': objective,
            'iterations': iteration[0],
            'convergence': result.func_vals if hasattr(result, 'func_vals') else []
        }
    
    def _optimize_grid(self, start_date: date, end_date: date, objective: str) -> Dict:
        """Grid search optimization"""
        print("[Optimizer] Using grid search")
        
        # Generate parameter grid
        param_grid = self._generate_param_grid()
        
        print(f"  Total combinations: {len(param_grid)}")
        
        best_score = -999.0
        best_params = None
        results_list = []
        
        for i, params in enumerate(param_grid):
            print(f"\n  Testing combination {i+1}/{len(param_grid)}")
            print(f"    Params: {params}")
            
            # Run backtest
            config_overrides = {
                'start_date': start_date,
                'end_date': end_date,
            }
            config_overrides.update(params)
            
            try:
                engine = self.engine_factory(config_overrides=config_overrides)
                results = engine.run()
                
                # Extract objective
                analysis = results['analysis']
                pm = analysis.get('portfolio_metrics', {})
                
                if objective == 'sharpe':
                    score = pm.get('sharpe_ratio', -999.0)
                    if score == 0.0:  # Invalid
                        score = -999.0
                elif objective == 'return':
                    score = pm.get('annual_return_pct', 0.0) / 100
                elif objective == 'calmar':
                    score = pm.get('calmar_ratio', 0.0)
                else:
                    score = pm.get('sharpe_ratio', 0.0)
                
                print(f"    {objective} = {score:.4f}")
                
                results_list.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"    ✓ NEW BEST!")
            
            except Exception as e:
                print(f"    FAILED: {e}")
        
        print(f"\n[Optimizer] Grid search complete!")
        print(f"  Best {objective}: {best_score:.4f}")
        print(f"  Best parameters: {json.dumps(best_params, indent=2)}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'objective': objective,
            'iterations': len(param_grid),
            'all_results': results_list
        }
    
    def _optimize_random(self, start_date: date, end_date: date, objective: str) -> Dict:
        """Random search optimization"""
        print("[Optimizer] Using random search")
        
        max_iter = self.config.get('max_iterations', 50)
        
        best_score = -999.0
        best_params = None
        results_list = []
        
        for i in range(max_iter):
            # Sample random parameters
            params = self._sample_random_params()
            
            print(f"\n  Iteration {i+1}/{max_iter}")
            print(f"    Params: {params}")
            
            # Run backtest
            config_overrides = {
                'start_date': start_date,
                'end_date': end_date,
            }
            config_overrides.update(params)
            
            try:
                engine = self.engine_factory(config_overrides=config_overrides)
                results = engine.run()
                
                # Extract objective
                analysis = results['analysis']
                pm = analysis.get('portfolio_metrics', {})
                
                if objective == 'sharpe':
                    score = pm.get('sharpe_ratio', -999.0)
                    if score == 0.0:  # Invalid
                        score = -999.0
                elif objective == 'return':
                    score = pm.get('annual_return_pct', 0.0) / 100
                elif objective == 'calmar':
                    score = pm.get('calmar_ratio', 0.0)
                else:
                    score = pm.get('sharpe_ratio', 0.0)
                
                print(f"    {objective} = {score:.4f}")
                
                results_list.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"    ✓ NEW BEST!")
            
            except Exception as e:
                print(f"    FAILED: {e}")
        
        print(f"\n[Optimizer] Random search complete!")
        print(f"  Best {objective}: {best_score:.4f}")
        print(f"  Best parameters: {json.dumps(best_params, indent=2)}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'objective': objective,
            'iterations': max_iter,
            'all_results': results_list
        }
    
    def _generate_param_grid(self) -> List[Dict]:
        """Generate grid of all parameter combinations"""
        import itertools
        
        # Convert param_space to lists of values
        param_lists = {}
        for name, space in self.param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                # Continuous: sample 5 points
                param_lists[name] = np.linspace(space[0], space[1], 5).tolist()
            elif isinstance(space, list):
                param_lists[name] = space
            else:
                param_lists[name] = [space]
        
        # Generate all combinations
        keys = list(param_lists.keys())
        values = [param_lists[k] for k in keys]
        
        combinations = list(itertools.product(*values))
        
        grid = []
        for combo in combinations:
            grid.append(dict(zip(keys, combo)))
        
        return grid
    
    def _sample_random_params(self) -> Dict:
        """Sample random parameters from search space"""
        params = {}
        
        for name, space in self.param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                # Continuous: uniform random
                params[name] = np.random.uniform(space[0], space[1])
            elif isinstance(space, list):
                # Categorical: random choice
                params[name] = np.random.choice(space)
            else:
                params[name] = space
        
        return params
    
    def walk_forward_validate(self,
                             start_date: date,
                             end_date: date,
                             train_window: int = 180,
                             test_window: int = 30,
                             step_size: int = 30) -> Dict:
        """
        Walk-forward validation
        
        Args:
            start_date: Start of validation period
            end_date: End of validation period
            train_window: Training window in days
            test_window: Testing window in days
            step_size: Step size in days
        
        Returns:
            dict with validation results
        """
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD VALIDATION")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Train window: {train_window} days")
        print(f"Test window: {test_window} days")
        print(f"Step size: {step_size} days")
        print("")
        
        current_date = start_date + timedelta(days=train_window)
        fold = 1
        
        validation_results = []
        
        while current_date + timedelta(days=test_window) <= end_date:
            train_start = current_date - timedelta(days=train_window)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_window)
            
            print(f"\nFold {fold}:")
            print(f"  Train: {train_start} to {train_end}")
            print(f"  Test:  {test_start} to {test_end}")
            
            # Optimize on training set
            print(f"  Optimizing on training set...")
            opt_result = self.optimize(train_start, train_end, objective='sharpe')
            best_params = opt_result['best_params']
            
            # Validate on test set
            print(f"  Validating on test set...")
            config_overrides = {
                'start_date': test_start,
                'end_date': test_end,
            }
            config_overrides.update(best_params)
            
            try:
                engine = self.engine_factory(config_overrides=config_overrides)
                results = engine.run()
                
                analysis = results['analysis']
                pm = analysis.get('portfolio_metrics', {})
                
                validation_results.append({
                    'fold': fold,
                    'train_start': str(train_start),
                    'train_end': str(train_end),
                    'test_start': str(test_start),
                    'test_end': str(test_end),
                    'best_params': best_params,
                    'test_sharpe': pm.get('sharpe_ratio', 0.0),
                    'test_return': pm.get('total_return_pct', 0.0),
                    'test_max_dd': pm.get('max_drawdown_pct', 0.0),
                })
                
                print(f"  Test Sharpe: {pm.get('sharpe_ratio', 0.0):.3f}")
                print(f"  Test Return: {pm.get('total_return_pct', 0.0):.2f}%")
            
            except Exception as e:
                print(f"  FAILED: {e}")
            
            current_date += timedelta(days=step_size)
            fold += 1
        
        # Summary
        df_results = pd.DataFrame(validation_results)
        
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total folds: {len(validation_results)}")
        print(f"\nAverage test performance:")
        print(f"  Sharpe:  {df_results['test_sharpe'].mean():.3f}")
        print(f"  Return:  {df_results['test_return'].mean():.2f}%")
        print(f"  Max DD:  {df_results['test_max_dd'].mean():.2f}%")
        
        return {
            'validation_results': validation_results,
            'summary': {
                'avg_test_sharpe': df_results['test_sharpe'].mean(),
                'avg_test_return': df_results['test_return'].mean(),
                'avg_test_max_dd': df_results['test_max_dd'].mean(),
            }
        }

