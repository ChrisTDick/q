import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MarketEnvironment:
    """
    Simplified multi-agent market environment simulating an auction-like mechanism.

    Takers submit market orders, and multiple makers provide liquidity at prices
    determined by their actions relative to the current market price.
    """
    def __init__(self, num_makers, initial_price=100.0, volatility=0.05,
                 taker_volume_range=(50, 200), # Increased taker volume for multi-agent
                 maker_liquidity_per_agent=50, tick_size=0.01, max_steps=50): # Added max_steps
        """
        Initializes the multi-agent market environment.

        Args:
            num_makers (int): The number of competing maker agents.
            initial_price (float): Starting price of the asset.
            volatility (float): The volatility of the underlying asset price.
            taker_volume_range (tuple): Range of volumes for taker orders.
            maker_liquidity_per_agent (int): Max liquidity EACH maker can provide per step.
            tick_size (float): Minimum price increment.
            max_steps (int): Maximum number of time steps per episode.
        """
        self.num_makers = num_makers
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = volatility
        self.taker_volume_range = taker_volume_range
        self.maker_liquidity_per_agent = maker_liquidity_per_agent
        self.tick_size = tick_size
        self.max_steps = max_steps # Store max steps
        self.time = 0

    def reset(self):
        """Resets the market to the initial state for a new episode."""
        self.current_price = self.initial_price
        self.time = 0
        # Note: State representation is simplified (only current price).
        # A more complex state could include order book depth, recent volume, etc.
        return self.current_price

    def step(self, maker_actions):
        """
        Simulates one step of the market interaction with multiple makers.

        Args:
            maker_actions (dict): A dictionary {maker_id: action}, where action is
                                  the price level relative to the current price.

        Returns:
            tuple: (next_state, rewards, done, info)
                   - next_state: The next market price.
                   - rewards (dict): {maker_id: reward} for each maker.
                   - done: Whether the episode is finished.
                   - info: Additional information (Taker PNL, total executed volume, etc.).
        """
        self.time += 1

        # --- 1. Taker Arrives ---
        taker_volume_to_sell = random.randint(*self.taker_volume_range)
        taker_strategy = random.choice(['aggressive', 'passive']) # Simplified taker strategy

        # --- 2. Makers Provide Liquidity (Bids) ---
        # Collect bids from all makers
        # bids = {price_level: {maker_id: liquidity}}
        bids = defaultdict(lambda: defaultdict(int))
        maker_prices = {} # Store the price each maker offered
        for maker_id, action in maker_actions.items():
            # Action is relative ticks from current price. Higher action = higher bid price.
            bid_price = self.current_price + action * self.tick_size
            maker_prices[maker_id] = bid_price
            # Assume each maker posts their full liquidity at their chosen price
            bids[bid_price][maker_id] += self.maker_liquidity_per_agent

        # --- 3. Market Order Execution (Taker Sells) ---
        executed_volume_total = 0
        executed_volume_per_maker = defaultdict(int)
        taker_pnl = 0.0
        volume_remaining = taker_volume_to_sell

        # Sort bids by price (highest price first)
        sorted_bid_prices = sorted(bids.keys(), reverse=True)

        execution_prices_volumes = [] # Store (price, volume) for avg calculation

        for price in sorted_bid_prices:
            if volume_remaining <= 0:
                break

            # Makers offering liquidity at this price level
            makers_at_price = list(bids[price].keys())
            random.shuffle(makers_at_price) # Randomize priority among makers at same price

            for maker_id in makers_at_price:
                if volume_remaining <= 0:
                    break

                liquidity_offered = bids[price][maker_id]
                volume_to_execute = min(volume_remaining, liquidity_offered)

                executed_volume_per_maker[maker_id] += volume_to_execute
                executed_volume_total += volume_to_execute
                volume_remaining -= volume_to_execute
                execution_prices_volumes.append((price, volume_to_execute))

        # Calculate average execution price for the taker
        if executed_volume_total > 0:
            avg_execution_price = sum(p * v for p, v in execution_prices_volumes) / executed_volume_total
            taker_pnl = (self.initial_price - avg_execution_price) * executed_volume_total
        else:
            avg_execution_price = self.current_price # No execution happened
            taker_pnl = 0

        # --- 4. Calculate Price Impact and Next State ---
        price_impact_factor = 0.0
        if taker_strategy == 'aggressive':
            price_impact_factor = 0.1 * self.volatility * (executed_volume_total / self.taker_volume_range[1])
        elif taker_strategy == 'passive':
             price_impact_factor = 0.02 * self.volatility * (executed_volume_total / self.taker_volume_range[1])

        # Price moves down based on executed sell volume
        self.current_price *= (1 - price_impact_factor)
        next_state = self.current_price # Simplified state

        # --- 5. Calculate Maker Rewards ---
        maker_rewards = {}
        # Reward is proportional negative share of taker PNL
        total_taker_loss = -taker_pnl
        for maker_id in maker_actions.keys():
             if executed_volume_total > 0:
                 proportion_executed = executed_volume_per_maker[maker_id] / executed_volume_total
                 maker_rewards[maker_id] = proportion_executed * total_taker_loss
             else:
                 maker_rewards[maker_id] = 0 # No reward if no volume executed

        # --- 6. Check if Done ---
        done = self.time >= self.max_steps

        info = {
            'taker_pnl': taker_pnl,
            'executed_volume_total': executed_volume_total,
            'executed_volume_per_maker': executed_volume_per_maker,
            'avg_execution_price': avg_execution_price,
            'maker_prices': maker_prices
        }

        # Return state (same for all agents), rewards (dict), done, info
        return next_state, maker_rewards, done, info


class MakerAgent:
    """
    Market maker agent using Q-learning to optimize its bid placement strategy.
    (Largely unchanged, but interacts within a multi-agent context)
    """
    def __init__(self, agent_id, action_space=[-2, -1, 0, 1, 2], learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01): # Added min exploration
        """
        Initializes the maker agent.

        Args:
            agent_id (int): Unique identifier for the agent.
            action_space (list): Possible price levels relative to the current price.
            learning_rate (float): Learning rate for Q-learning.
            discount_factor (float): Discount factor for future rewards.
            exploration_rate (float): Initial exploration rate.
            exploration_decay (float): Decay factor for exploration rate.
            min_exploration_rate (float): Minimum value for exploration rate.
        """
        self.agent_id = agent_id
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        # Q-table: {(state_discrete, action): q_value}
        # State needs discretization for tabular Q-learning
        self.q_table = defaultdict(float)

    def _discretize_state(self, state, num_bins=10, price_range=(95, 105)):
        """Discretizes the continuous price state into bins."""
        # Simple binning - more sophisticated methods could be used
        if state <= price_range[0]:
            return 0
        if state >= price_range[1]:
            return num_bins - 1
        bin_size = (price_range[1] - price_range[0]) / num_bins
        return int((state - price_range[0]) // bin_size)

    def get_q_value(self, state, action):
        """Retrieves the Q-value for a given discretized state-action pair."""
        discrete_state = self._discretize_state(state)
        return self.q_table[(discrete_state, action)]

    def update_q_value(self, state, action, next_state, reward):
        """Updates the Q-value based on the Q-learning update rule."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        old_q_value = self.get_q_value(state, action) # Uses discretized state internally
        next_max_q = max(self.get_q_value(next_state, a) for a in self.action_space) # Uses discretized state internally
        new_q_value = (1 - self.learning_rate) * old_q_value + \
                       self.learning_rate * (reward + self.discount_factor * next_max_q)
        self.q_table[(discrete_state, action)] = new_q_value

    def get_action(self, state):
        """
        Selects an action based on an epsilon-greedy policy using discretized state.

        Args:
            state (float): The current market price (continuous).

        Returns:
            int: The chosen action (price level relative to current price).
        """
        discrete_state = self._discretize_state(state) # Discretize for Q-table lookup

        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)  # Explore
        else:
            # Exploit: Find action with highest Q-value for this discrete state
            q_values = [self.q_table.get((discrete_state, action), 0.0) for action in self.action_space]
            # Handle ties randomly
            max_q = max(q_values)
            best_actions = [self.action_space[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update_exploration_rate(self):
        """Decays the exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


def train_and_evaluate(env, maker_agents, num_episodes=2000): # Increased episodes
    """
    Trains multiple maker agents in the market environment and evaluates performance.

    Args:
        env (MarketEnvironment): The market simulation environment.
        maker_agents (list): A list of MakerAgent instances.
        num_episodes (int): Number of training episodes.
    """
    num_makers = len(maker_agents)
    # Store metrics per agent and overall
    all_rewards_per_agent = [[] for _ in range(num_makers)]
    all_taker_pnls = []
    all_avg_exec_prices = []
    total_rewards_per_episode = [] # Sum of rewards for all makers

    print(f"Starting training with {num_makers} makers for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards_per_agent = defaultdict(float)
        episode_taker_pnl = 0
        episode_exec_prices = []
        done = False

        while not done:
            # 1. Get actions from all agents
            actions = {agent.agent_id: agent.get_action(state) for agent in maker_agents}

            # 2. Environment steps
            next_state, rewards, done, info = env.step(actions) # Env handles multi-agent step

            # 3. Update each agent
            for agent in maker_agents:
                agent_reward = rewards[agent.agent_id]
                agent_action = actions[agent.agent_id]
                agent.update_q_value(state, agent_action, next_state, agent_reward)
                episode_rewards_per_agent[agent.agent_id] += agent_reward

            state = next_state
            episode_taker_pnl += info['taker_pnl']
            if info['executed_volume_total'] > 0:
                episode_exec_prices.append(info['avg_execution_price'])

        # End of episode updates
        for i, agent in enumerate(maker_agents):
            agent.update_exploration_rate()
            all_rewards_per_agent[i].append(episode_rewards_per_agent[agent.agent_id])

        all_taker_pnls.append(episode_taker_pnl)
        total_rewards_per_episode.append(sum(episode_rewards_per_agent.values()))
        if episode_exec_prices:
             all_avg_exec_prices.append(np.mean(episode_exec_prices))
        else:
             all_avg_exec_prices.append(env.initial_price) # Append initial if no trades

        # Print progress
        if (episode + 1) % 100 == 0:
             avg_total_reward = np.mean(total_rewards_per_episode[-100:])
             avg_taker_pnl = np.mean(all_taker_pnls[-100:])
             print(f"Episode {episode + 1}/{num_episodes}: Avg Total Reward (100 ep): {avg_total_reward:.2f}, "
                   f"Avg Taker PNL (100 ep): {avg_taker_pnl:.2f}, "
                   f"Exploration Rate (Agent 0): {maker_agents[0].exploration_rate:.3f}")

    print("\nTraining complete.")
    avg_final_rewards = [np.mean(rewards) for rewards in all_rewards_per_agent]
    print(f"Average Final Reward per Agent: {[f'{r:.2f}' for r in avg_final_rewards]}")
    print(f"Overall Average Taker PNL: {np.mean(all_taker_pnls):.2f}")

    # --- Visualization ---
    plt.figure(figsize=(15, 10))

    # Plot 1: Total rewards per episode (sum of all makers)
    plt.subplot(2, 2, 1)
    # Calculate moving average for smoother plot
    window_size = 50
    if len(total_rewards_per_episode) >= window_size:
        moving_avg_rewards = np.convolve(total_rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg_rewards, label=f'{window_size}-Ep Moving Avg')
    plt.plot(total_rewards_per_episode, alpha=0.3, label='Raw Total Reward')
    plt.title('Total Maker Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (All Makers)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Taker PNL per episode
    plt.subplot(2, 2, 2)
    if len(all_taker_pnls) >= window_size:
        moving_avg_taker_pnl = np.convolve(all_taker_pnls, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg_taker_pnl, label=f'{window_size}-Ep Moving Avg')
    plt.plot(all_taker_pnls, alpha=0.3, label='Raw Taker PNL')
    plt.title('Taker PNL per Episode')
    plt.xlabel('Episode')
    plt.ylabel('PNL')
    plt.legend()
    plt.grid(True)


    # Plot 3: Average Execution Price per episode
    plt.subplot(2, 2, 3)
    if len(all_avg_exec_prices) >= window_size:
        moving_avg_exec_price = np.convolve(all_avg_exec_prices, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg_exec_price, label=f'{window_size}-Ep Moving Avg')
    plt.plot(all_avg_exec_prices, alpha=0.3, label='Raw Avg Exec Price')
    plt.title('Average Execution Price per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Price')
    plt.axhline(env.initial_price, color='r', linestyle='--', label='Initial Price')
    plt.legend()
    plt.grid(True)

    # Plot 4: Exploration Rate Decay (Agent 0)
    plt.subplot(2, 2, 4)
    exploration_rates = [agent.min_exploration_rate + (1.0 - agent.min_exploration_rate) * (agent.exploration_decay**ep) for ep in range(num_episodes)]
    plt.plot(exploration_rates)
    plt.title('Exploration Rate Decay (Example)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)


    plt.tight_layout()
    plt.show()

    # Optional: Print Q-table sample for one agent
    print("\nSample Q-Table entries for Agent 0:")
    count = 0
    for (state, action), q_value in maker_agents[0].q_table.items():
        if count < 10: # Print first 10 entries
            print(f"  State (Discrete Bin): {state}, Action: {action}, Q-Value: {q_value:.3f}")
            count += 1
        else:
            break


if __name__ == "__main__":
    # --- Simulation Parameters ---
    NUM_MAKERS = 3
    MAX_STEPS_PER_EPISODE = 100 # Increased time steps
    NUM_EPISODES = 5000        # Increased episodes for better learning

    # --- Initialize Environment ---
    env = MarketEnvironment(num_makers=NUM_MAKERS, max_steps=MAX_STEPS_PER_EPISODE)

    # --- Initialize Maker Agents ---
    # Define action space common to all agents
    # More negative action = lower bid price (further from current price)
    # More positive action = higher bid price (closer to current price)
    action_space = [-5, -3, -1, 0, 1] # Example action space (relative ticks)
    maker_agents = [MakerAgent(agent_id=i, action_space=action_space) for i in range(NUM_MAKERS)]

    # --- Train and Evaluate ---
    train_and_evaluate(env, maker_agents, num_episodes=NUM_EPISODES)
