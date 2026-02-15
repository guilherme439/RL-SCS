import pytest
import random
from multiprocessing import Pool, cpu_count

from .base_test import BaseTest
from rl_scs.SCS_Game import SCS_Game


def run_stress_cycles(args):
    """
    Worker function to run stress test cycles.
    Separated for multiprocessing compatibility.
    
    Args:
        args: Tuple of (config_path, seed, num_cycles)
    
    Returns:
        Tuple of (run_number, success, error_message)
    """
    config_path, seed, num_cycles, run_num = args
    
    try:
        env = SCS_Game(config_path)
        
        for cycle in range(num_cycles):
            env.reset(seed=seed + cycle)
            
            # Play a short game
            for agent in env.agent_iter(max_iter=50):
                _, _, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                else:
                    # Sample directly from legal actions for efficiency
                    action_mask = info.get("action_mask")
                    if action_mask is not None:
                        legal_actions = [i for i, legal in enumerate(action_mask) if legal]
                        action = random.choice(legal_actions) if legal_actions else None
                    else:
                        action = None
                
                env.step(action)
                
                # Break if game is over
                if not env.agents:
                    break
        
        return (run_num, True, None)
    
    except Exception as e:
        return (run_num, False, str(e))


class TestStress(BaseTest):
    
    @pytest.mark.stress
    @pytest.mark.parametrize("config_name", BaseTest.configs)
    def test_stress_multiple_runs(self, config_name):
        """
        Stress test each config with 5 runs of 400 cycles each (2000 total),
        using different seeds and parallel workers for efficiency
        """
        print(f"\nStress testing (5 runs × 400 cycles, {cpu_count()} workers): {config_name}")
        
        config_path = self.get_config_path(config_name)
        num_runs = 5
        cycles_per_run = 400
        num_cpus = num_runs
        base_seed = 42
        
        # Prepare arguments for parallel execution
        worker_args = [
            (config_path, base_seed + run, cycles_per_run, run + 1)
            for run in range(num_runs)
        ]
        
        # Run in parallel using process pool
        with Pool(processes=min(num_cpus, cpu_count())) as pool:
            results = pool.map(run_stress_cycles, worker_args)
        
        # Check results and report
        for run_num, success, error in results:
            if success:
                print(f"  Run {run_num}/{num_runs}: ✓")
            else:
                print(f"  Run {run_num}/{num_runs}: ✗ - {error}")
                pytest.fail(f"Stress test run {run_num} failed: {error}")
        
        print(f"  Total: {num_runs * cycles_per_run} cycles completed successfully")
