import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility.
random.seed(42)
np.random.seed(42)

# -------------------------------------
# Global parameters and configuration
# -------------------------------------
INITIAL_PRICE = 100.0
TICK_SIZE = 0.1
BASE_QUANTITY = 100  # default starting quantity for maker orders
QUANTITY_INCREMENT = 10

MAX_TIME = 50  # number of discrete time steps in a game
TOTAL_TAKER_VOLUME = 8000  # total volume taker intends to execute during the game

# Q-learning parameters for makers:
Q_LEARNING_PARAMS = {
    'alpha': 0.1,    # learning rate
    'gamma': 0.95,   # discount factor
    'epsilon': 0.2   # exploration rate
}

# Action mapping for makers (discrete actions)
ACTION_SPACE = {
    0: "increase_bid_price",    # Increase bid price by one tick
    1: "decrease_bid_price",    # Decrease bid price by one tick
    2: "maintain_bid_price",    # Maintain current bid price
    3: "increase_quantity",     # Increase bid quantity by QUANTITY_INCREMENT
    4: "decrease_quantity",     # Decrease bid quantity by QUANTITY_INCREMENT
    5: "maintain_quantity",     # Maintain current bid quantity
    6: "aggressive_replenish"   # Aggressively replenish bid quantity to BASE_QUANTITY * 2
}

# -------------------------------------
# Market Environment: Central Limit Order Book
# -------------------------------------
class MarketEnvironment:
    def __init__(self, initial_price, max_time):
        self.initial_price = initial_price
        self.current_time = 0
        self.max_time = max_time
        # The order book will be a list of maker orders for each time step.
        self.maker_orders = []  # each element is a tuple: (maker_id, bid_price, quantity)
    
    def set_maker_orders(self, orders):
        """Update the order book with maker orders.
        
        orders: list of (maker_id, bid_price, quantity)
        """
        self.maker_orders = orders

    def process_taker_order(self, taker_order_volume):
        """
        Process the taker's sell order.
        
        Matching process:
         - Sort maker orders by bid price descending (taker will be incentivized to execute against the highest bid first).
         - Fill the taker order until the full volume is met or the order book is exhausted.
        
        Returns:
         - executed: a dict mapping maker_id to executed volume.
         - avg_exec_price: the weighted average execution price.
        """
        # Sort orders by bid_price descending
        orders = sorted(self.maker_orders, key=lambda x: x[1], reverse=True)
        
        remaining_volume = taker_order_volume
        executed = {}
        total_price_volume = 0.0  # accumulate price * volume for computing weighted average
        
        for maker_id, bid_price, quantity in orders:
            if remaining_volume <= 0:
                break
            executed_volume = min(quantity, remaining_volume)
            executed[maker_id] = executed.get(maker_id, 0) + executed_volume
            total_price_volume += bid_price * executed_volume
            remaining_volume -= executed_volume
        
        filled_volume = taker_order_volume - remaining_volume
        avg_exec_price = total_price_volume / filled_volume if filled_volume > 0 else self.initial_price
        return executed, avg_exec_price

# -------------------------------------
# Taker: Liquidity Consumer Agent
# -------------------------------------
class Taker:
    def __init__(self, strategy, total_volume, max_time):
        """
        strategy: string indicating which strategy to employ ("aggressive", "passive", "reactive")
        total_volume: total volume the taker must execute over the game
        max_time: total number of time steps
        """
        self.strategy = strategy
        self.total_volume = total_volume
        self.max_time = max_time
        self.remaining_volume = total_volume
        
    def get_order_volume(self, current_time):
        """
        Determine the volume to execute at the current time step based on strategy.
        """
        if self.strategy == "aggressive":
            # Aggressive: execute a larger chunk each time.
            volume = int(self.total_volume / self.max_time * 1.5)
        elif self.strategy == "passive":
            # Passive: execute a conservative fraction.
            volume = int(self.total_volume / self.max_time * 0.75)
        elif self.strategy == "reactive":
            # Reactive: could be expanded to incorporate market feedback.
            volume = int(self.total_volume / self.max_time)
        else:
            volume = int(self.total_volume / self.max_time)
        
        # Ensure not to exceed the remaining volume
        volume = min(volume, self.remaining_volume)
        self.remaining_volume -= volume
        return volume

# -------------------------------------
# Maker: Market Making Agent using Q-Learning
# -------------------------------------
class Maker:
    def __init__(self, maker_id, initial_bid, initial_quantity, initial_price, tick_size, quantity_increment, q_learning_params):
        self.maker_id = maker_id
        self.bid_price = initial_bid
        self.quantity = initial_quantity
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.quantity_increment = quantity_increment
        
        # Q-learning parameters
        self.alpha = q_learning_params.get('alpha', 0.1)
        self.gamma = q_learning_params.get('gamma', 0.95)
        self.epsilon = q_learning_params.get('epsilon', 0.2)
        
        # Q-table: keys are state tuples and values are lists (one per action) of Q-values.
        self.q_table = {}
        
        # Store last state and action to update later.
        self.last_state = None
        self.last_action = None
        
        self.cumulative_reward = 0  # to track performance over a game
    
    def get_state(self, current_time, total_time):
        """
        Create a simplified state representation.
          - time_remaining: discretized (e.g. high if >50% time steps remain, low otherwise)
          - price_offset: difference (in ticks) from the initial price.
          - quantity_level: 0 if quantity is below BASE_QUANTITY, 1 otherwise.
        """
        time_remaining = total_time - current_time
        time_state = 1 if time_remaining > total_time * 0.5 else 0  # 1: early, 0: late
        
        # Price offset in ticks:
        price_offset = int(round((self.bid_price - self.initial_price) / self.tick_size))
        
        quantity_level = 1 if self.quantity >= BASE_QUANTITY else 0
        
        return (time_state, price_offset, quantity_level)
    
    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        """
        # Initialize Q-values for unseen state.
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(len(ACTION_SPACE))]
        
        # Epsilon-greedy: explore with probability epsilon.
        if random.random() < self.epsilon:
            action = random.choice(list(ACTION_SPACE.keys()))
        else:
            # Exploit: choose the action with the highest Q-value.
            q_values = self.q_table[state]
            max_q = max(q_values)
            # In case several actions have equal value, randomly choose among them.
            candidate_actions = [action for action, q in enumerate(q_values) if q == max_q]
            action = random.choice(candidate_actions)
        return action
    
    def update_order(self, action):
        """
        Update bid price and quantity according to the chosen action.
        """
        if ACTION_SPACE[action] == "increase_bid_price":
            self.bid_price += self.tick_size
        elif ACTION_SPACE[action] == "decrease_bid_price":
            self.bid_price -= self.tick_size
        elif ACTION_SPACE[action] == "maintain_bid_price":
            pass
        elif ACTION_SPACE[action] == "increase_quantity":
            self.quantity += self.quantity_increment
        elif ACTION_SPACE[action] == "decrease_quantity":
            # Ensure quantity does not drop below a minimum (say, 1)
            self.quantity = max(1, self.quantity - self.quantity_increment)
        elif ACTION_SPACE[action] == "maintain_quantity":
            pass
        elif ACTION_SPACE[action] == "aggressive_replenish":
            # Aggressively replenish to a higher quantity.
            self.quantity = BASE_QUANTITY * 2
        
        # For safety, enforce that bid_price remains reasonable.
        if self.bid_price < self.initial_price - 5:  # do not be too far below the initial
            self.bid_price = self.initial_price - 5
        if self.bid_price > self.initial_price + 5:   # likewise, not too high above
            self.bid_price = self.initial_price + 5
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for the state-action pair using Q-learning.
        """
        # Initialize Q-values for state and next_state if necessary.
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(len(ACTION_SPACE))]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0 for _ in range(len(ACTION_SPACE))]
        
        max_future_q = max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        # Standard Q-learning update.
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

# -------------------------------------
# Simulation driver: run a single game (episode)
# -------------------------------------
def run_simulation(num_games=50, num_makers=2, taker_strategy="aggressive"):
    """
    Run multiple simulation games and record maker performance.
    
    num_games: number of episodes/games to simulate.
    num_makers: how many maker agents are competing.
    taker_strategy: strategy deployed by the taker ("aggressive", "passive", "reactive").
    
    Returns:
     - maker_rewards: a list of average cumulative rewards per maker for each game.
    """
    maker_rewards = {m: [] for m in range(num_makers)}
    
    for game in range(num_games):
        # Initialize environment, taker, and maker agents.
        env = MarketEnvironment(INITIAL_PRICE, MAX_TIME)
        taker = Taker(taker_strategy, TOTAL_TAKER_VOLUME, MAX_TIME)
        
        makers = []
        for m in range(num_makers):
            # For initial conditions, let makers start with a bid slightly below the initial price and base quantity.
            initial_bid = INITIAL_PRICE - TICK_SIZE  
            maker = Maker(m, initial_bid, BASE_QUANTITY, INITIAL_PRICE, TICK_SIZE, QUANTITY_INCREMENT, Q_LEARNING_PARAMS)
            makers.append(maker)
        
        # Run the game over discrete time steps.
        for t in range(MAX_TIME):
            # Makers update their orders based on their state and policy.
            orders = []
            for maker in makers:
                state = maker.get_state(t, MAX_TIME)
                action = maker.choose_action(state)
                maker.last_state = state
                maker.last_action = action
                maker.update_order(action)
                orders.append((maker.maker_id, maker.bid_price, maker.quantity))
            
            # Update the market with the current maker orders.
            env.set_maker_orders(orders)
            
            # Taker places an order.
            taker_volume = taker.get_order_volume(t)
            executed, avg_exec_price = env.process_taker_order(taker_volume)
            
            # Compute maker rewards based on execution.
            # (If a maker is not executed, reward is 0 for this step.)
            for maker in makers:
                exec_volume = executed.get(maker.maker_id, 0)
                # Reward is proportional to maker’s fill fraction and the difference between the initial price and maker’s bid.
                if taker_volume > 0:
                    reward = (exec_volume / taker_volume) * (env.initial_price - maker.bid_price)
                else:
                    reward = 0.0
                maker.cumulative_reward += reward
                
                # Next state after orders are processed.
                next_state = maker.get_state(t + 1, MAX_TIME)
                maker.update_q_value(maker.last_state, maker.last_action, reward, next_state)
        
        # End of game: record the cumulative rewards for each maker.
        for maker in makers:
            maker_rewards[maker.maker_id].append(maker.cumulative_reward)
    
    return maker_rewards

# -------------------------------------
# Run the simulation and visualize outcomes.
# -------------------------------------
num_games = 100
num_makers = 3
taker_strategy = "aggressive"  # try "passive" or "reactive" to compare strategies

maker_rewards = run_simulation(num_games=num_games, num_makers=num_makers, taker_strategy=taker_strategy)

# Plot the average cumulative reward trajectory for each maker over the games.
plt.figure(figsize=(10, 6))
for maker_id, rewards in maker_rewards.items():
    plt.plot(rewards, label=f"Maker {maker_id}")
plt.xlabel("Game number")
plt.ylabel("Cumulative Reward")
plt.title("Maker Performance over Games")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------
# Additional Visualizations:
# For example, show the average cumulative reward across makers.
avg_rewards = np.mean([maker_rewards[m] for m in maker_rewards], axis=0)
plt.figure(figsize=(8, 5))
plt.plot(avg_rewards, marker='o')
plt.xlabel("Game number")
plt.ylabel("Average Cumulative Reward")
plt.title("Average Maker Performance over Games")
plt.grid(True)
plt.show()
