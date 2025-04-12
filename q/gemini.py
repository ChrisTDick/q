import random
import numpy as np

class MarketEnvironment:
    """
    Simplified market environment simulating an auction-like mechanism.

    Instead of a full CLOB, this simulates a single-round interaction where
    takers submit market orders and makers provide liquidity at a set price.
    """
    def __init__(self, initial_price=100.0, volatility=0.05, taker_volume_range=(10, 100),
                 maker_liquidity=50, tick_size=0.01):
        """
        Initializes the market environment.

        Args:
            initial_price (float): Starting price of the asset.
            volatility (float):  The volatility of the underlying asset price.
            taker_volume_range (tuple): Range of volumes for taker orders.
            maker_liquidity (int):  Maximum liquidity provided by the maker.
            tick_size (float): Minimum price increment.
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = volatility
        self.taker_volume_range = taker_volume_range
        self.maker_liquidity = maker_liquidity
        self.tick_size = tick_size
        self.time = 0

    def reset(self):
        """Resets the market to the initial state."""
        self.current_price = self.initial_price
        self.time = 0
        return self.current_price

    def step(self, maker_action):
        """
        Simulates one step of the market interaction.

        Args:
            maker_action (int):  The maker's action (price level relative
                                 to the current price).

        Returns:
            tuple: (next_state, reward, done, info)
                   - next_state: The next market price.
                   - reward: The maker's reward.
                   - done: Whether the episode is finished.
                   - info: Additional information (Taker PNL, etc.).
        """
        self.time += 1
        # 1. Taker arrives with a market order.
        taker_volume = random.randint(*self.taker_volume_range)
        taker_strategy = random.choice(['aggressive', 'passive'])  # Simplified taker strategy

        # 2. Maker provides liquidity.
        maker_price = self.current_price + maker_action * self.tick_size
        available_liquidity = min(self.maker_liquidity, taker_volume)

        # 3. Market order execution
        if taker_strategy == 'aggressive':
            self.current_price *= (1 - 0.1 * self.volatility) # Large price impact.
            execution_price = maker_price #taker takes the maker's price
            executed_volume = available_liquidity
        elif taker_strategy == 'passive':
            self.current_price *= (1 - 0.02 * self.volatility) # smaller price impact
            execution_price = maker_price #taker takes the maker's price.
            executed_volume = available_liquidity // 2
        else: #should not happen, unless extended
            execution_price = maker_price
            executed_volume = available_liquidity

        # 4. Calculate PNL
        taker_pnl = (self.initial_price - execution_price) * executed_volume
        maker_reward = -taker_pnl  # Simplified: Maker's profit is the negative of taker's loss.

        done = self.time >= 10  # End after a few steps for simplicity
        next_state = self.current_price

        info = {
            'taker_pnl': taker_pnl,
            'executed_volume': executed_volume,
            'execution_price': execution_price
        }

        return next_state, maker_reward, done, info


class MakerAgent:
    """
    Market maker agent using Q-learning to optimize its bid placement strategy.
    """
    def __init__(self, action_space=[-2, -1, 0, 1, 2], learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.5, exploration_decay=0.99):
        """
        Initializes the maker agent.

        Args:
            action_space (list): Possible price levels relative to the current price.
            learning_rate (float): Learning rate for Q-learning.
            discount_factor (float): Discount factor for future rewards.
            exploration_rate (float): Initial exploration rate.
            exploration_decay (float): Decay factor for exploration rate.
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Q-table is a dictionary {(state, action): q_value}

    def get_q_value(self, state, action):
        """Retrieves the Q-value for a given state-action pair."""
        return self.q_table.get((state, action), 0.0)  # Default to 0 if not seen before.

    def update_q_value(self, state, action, next_state, reward):
        """Updates the Q-value based on the Q-learning update rule."""
        old_q_value = self.get_q_value(state, action)
        next_max_q = max(self.get_q_value(next_state, a) for a in self.action_space)
        new_q_value = (1 - self.learning_rate) * old_q_value + \
                       self.learning_rate * (reward + self.discount_factor * next_max_q)
        self.q_table[(state, action)] = new_q_value

    def get_action(self, state):
        """
        Selects an action based on an epsilon-greedy policy.

        Args:
            state (float): The current market price.

        Returns:
            int: The chosen action (price level relative to current price).
        """
        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)  # Explore
        else:
            q_values = [self.get_q_value(state, action) for action in self.action_space]
            best_action = self.action_space[np.argmax(q_values)]
            return best_action  # Exploit

    def update_exploration_rate(self):
        """Decays the exploration rate."""
        self.exploration_rate *= self.exploration_decay



def train_and_evaluate(env, maker_agent, num_episodes=1000):
    """
    Trains the maker agent in the market environment and evaluates performance.

    Args:
        env (MarketEnvironment): The market simulation environment.
        maker_agent (MakerAgent): The Q-learning maker agent.
        num_episodes (int): Number of training episodes.
    """
    all_rewards = []
    all_taker_pnls = []
    all_execution_prices = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = 0
        episode_taker_pnls = 0
        episode_execution_prices = []
        done = False

        while not done:
            action = maker_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            maker_agent.update_q_value(state, action, next_state, reward)

            state = next_state
            episode_rewards += reward
            episode_taker_pnls += info['taker_pnl']
            episode_execution_prices.append(info['execution_price'])

        maker_agent.update_exploration_rate()
        all_rewards.append(episode_rewards)
        all_taker_pnls.append(episode_taker_pnls)
        all_execution_prices.append(np.mean(episode_execution_prices))

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Avg Reward: {np.mean(all_rewards[-100:]):.2f}, "
                  f"Avg Taker PNL: {np.mean(all_taker_pnls[-100:]):.2f}, "
                  f"Exploration Rate: {maker_agent.exploration_rate:.2f}")

    print("\nTraining complete.")
    print(f"Average Maker Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Taker PNL: {np.mean(all_taker_pnls):.2f}")

    # Basic Analysis and Visualization (within the text output)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards)
    plt.title('Maker Agent Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(all_taker_pnls)
    plt.title('Taker PNL per Episode')
    plt.xlabel('Episode')
    plt.ylabel('PNL')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(all_execution_prices)
    plt.title('Average Execution Price per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Initialize the environment and agents.
    env = MarketEnvironment()
    maker_agent = MakerAgent()

    # Train and evaluate the maker agent.
    train_and_evaluate(env, maker_agent)
