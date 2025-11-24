"""PettingZoo API compliance tests"""
import pytest
from pettingzoo.test import api_test

from .base_test import BaseTest
from .. import SCS_Game


class TestPettingZoo(BaseTest):
    """Test PettingZoo API compliance across different configurations"""
    
    @pytest.mark.api
    @pytest.mark.parametrize("config_name", BaseTest.configs)
    def test_api_compliance(self, config_name):
        """Test PettingZoo API compliance with 100 cycles for each config"""
        print(f"\nTesting API compliance: {config_name}")
        config_path = self.get_config_path(config_name)
        env = SCS_Game(config_path)
        api_test(env, num_cycles=200, verbose_progress=False)
    
    @pytest.mark.api
    def test_random_agents(self, max_steps=100):
        """Test random agents playing a full game using action_space.sample()"""
        print("\n" + "="*80)
        print("Testing Random Agents")
        print("="*80)
        
        config_path = self.get_config_path("randomized_config_5.yml")
        env = SCS_Game(config_path)
        env.reset(seed=42)
        
        step_count = 0
        for agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                # Use action_space.sample() with action mask
                action_mask = info.get("action_mask")
                if action_mask is not None:
                    # Sample from valid actions only
                    legal_actions = [i for i, legal in enumerate(action_mask) if legal]
                    if legal_actions:
                        # Use the environment's action space to sample
                        action = env.action_space(agent).sample(mask=action_mask)
                    else:
                        print(f"Warning: No legal actions for agent {agent}")
                        action = None
                else:
                    print(f"Warning: No action mask for agent {agent}")
                    action = None
            
            env.step(action)
            if (step_count % 10 == 0):
                print(f"Step {step_count}, Agent: {agent}")
            step_count += 1

        print(f"\nGame finished after {step_count} steps")
        print(f"Winner: {env.get_winner()}")
        print("✓ Random agents test passed!\n")
