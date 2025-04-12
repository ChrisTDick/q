import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm

class OrderBook:
    """Central Limit Order Book (CLOB) implementation."""
    
    def __init__(self, initial_price=100.0, tick_size=0.01):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.tick_size = tick_size
        self.bids = {}  # price -> {maker_id: quantity}
        self.asks = {}  # price -> {maker_id: quantity}
        self.trades = []  # List of executed trades
        self.mid_price_history = [initial_price]
    
    def add_bid(self, maker_id, price, quantity):
        """Add a bid order to the book."""
        price = round(price / self.tick_size) * self.tick_size  # Ensure price is aligned with tick size
        
        if price not in self.bids:
            self.bids[price] = {}
        
        if maker_id in self.bids[price]:
            self.bids[price][maker_id] += quantity
        else:
            self.bids[price][maker_id] = quantity
    
    def add_ask(self, maker_id, price, quantity):
        """Add an ask order to the book."""
        price = round(price / self.tick_size) * self.tick_size  # Ensure price is aligned with tick size
        
        if price not in self.asks:
            self.asks[price] = {}
        
        if maker_id in self.asks[price]:
            self.asks[price][maker_id] += quantity
        else:
            self.asks[price][maker_id] = quantity
    
    def execute_market_sell(self, quantity):
        """Execute a market sell order and return the executed trades."""
        executed_trades = []
        remaining_quantity = quantity
        maker_executions = {}  # maker_id -> executed_quantity
        
        # Sort bids in descending order (highest price first)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        
        total_value = 0
        
        for bid_price, maker_quantities in sorted_bids:
            # Sort makers by their quantities (to ensure fair execution)
            makers = list(maker_quantities.items())
            random.shuffle(makers)  # Random tie-breaking
            
            for maker_id, maker_quantity in makers:
                if remaining_quantity <= 0:
                    break
                
                executed_quantity = min(maker_quantity, remaining_quantity)
                remaining_quantity -= executed_quantity
                
                # Update the order book
                self.bids[bid_price][maker_id] -= executed_quantity
                if self.bids[bid_price][maker_id] <= 0:
                    del self.bids[bid_price][maker_id]
                
                if len(self.bids[bid_price]) == 0:
                    del self.bids[bid_price]
                
                # Record the trade
                trade = {
                    'price': bid_price,
                    'quantity': executed_quantity,
                    'maker_id': maker_id,
                    'side': 'sell'
                }
                executed_trades.append(trade)
                
                # Update maker executions
                if maker_id not in maker_executions:
                    maker_executions[maker_id] = 0
                maker_executions[maker_id] += executed_quantity
                
                # Update total value
                total_value += bid_price * executed_quantity
        
        # Update current price to the last executed price if any trades occurred
        if executed_trades:
            self.current_price = executed_trades[-1]['price']
            self.mid_price_history.append(self.current_price)
        
        # Calculate average execution price
        avg_execution_price = total_value / (quantity - remaining_quantity) if quantity - remaining_quantity > 0 else None
        
        return {
            'trades': executed_trades,
            'remaining_quantity': remaining_quantity,
            'maker_executions': maker_executions,
            'avg_execution_price': avg_execution_price
        }
    
    def get_best_bid(self):
        """Get the best (highest) bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        """Get the best (lowest) ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_bid_depth(self, levels=5):
        """Get the bid depth at the top 'levels' price levels."""
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:levels]
        return {price: sum(quantities.values()) for price, quantities in sorted_bids}
    
    def get_ask_depth(self, levels=5):
        """Get the ask depth at the top 'levels' price levels."""
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]
        return {price: sum(quantities.values()) for price, quantities in sorted_asks}
    
    def get_market_state(self, levels=3):
        """Get the current market state for RL state representation."""
        bid_depths = self.get_bid_depth(levels)
        bid_prices = sorted(bid_depths.keys(), reverse=True)
        bid_quantities = [bid_depths[p] for p in bid_prices]
        
        # Pad with zeros if not enough levels
        while len(bid_prices) < levels:
            bid_prices.append(0)
            bid_quantities.append(0)
        
        return {
            'current_price': self.current_price,
            'bid_prices': bid_prices[:levels],
            'bid_quantities': bid_quantities[:levels],
            'depth': len(self.bids)
        }
    
    def get_recent_trading_volume(self, lookback=10):
        """Calculate recent trading volume."""
        recent_trades = self.trades[-lookback:] if len(self.trades) >= lookback else self.trades
        return sum(trade['quantity'] for trade in recent_trades)
    
    def get_recent_trading_velocity(self, lookback=10):
        """Calculate recent trading velocity (change in price)."""
        if len(self.mid_price_history) < 2:
            return 0
        
        recent_prices = self.mid_price_history[-lookback:] if len(self.mid_price_history) >= lookback else self.mid_price_history
        if len(recent_prices) < 2:
            return 0
        
        return (recent_prices[-1] - recent_prices[0]) / len(recent_prices)


class QLearningAgent:
    """Q-Learning agent for market-making."""
    
    def __init__(self, agent_id, action_space, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        self.agent_id = agent_id
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        self.q_table = {}
        
        # Agent state
        self.current_bid_price = None
        self.current_bid_quantity = None
        self.inventory = 0
        self.cash = 0
        self.pnl_history = []
        self.executed_volume = 0
    
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        # Discretize continuous values
        price_bin = int(state['current_price'] * 100)  # 2 decimal places
        depth_bin = min(state['depth'], 10)  # Cap at 10 levels for simplicity
        
        # Create a tuple that can be used as a dictionary key
        bid_prices_tuple = tuple([int(p * 100) for p in state['bid_prices']])
        bid_quantities_tuple = tuple([min(q // 100, 10) for q in state['bid_quantities']])
        
        return (price_bin, depth_bin, bid_prices_tuple, bid_quantities_tuple)
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        
        # Explore: choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        
        # Exploit: choose the best action from Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        
        return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning algorithm."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if they don't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_space))
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        if done:
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state_key][action] = new_q
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
    
    def interpret_action(self, action_idx, current_price, tick_size=0.01, base_quantity=100):
        """Interpret the action index and return the corresponding order parameters."""
        # Define actions:
        # 0: Increase bid price by one tick
        # 1: Decrease bid price by one tick
        # 2: Maintain current bid price
        # 3: Increase bid quantity by a fixed increment
        # 4: Decrease bid quantity by a fixed increment
        # 5: Maintain current bid quantity
        # 6: Aggressively replenish bids at the current price
        
        # Initialize with default values
        if self.current_bid_price is None:
            self.current_bid_price = current_price - tick_size  # Start slightly below market
        
        if self.current_bid_quantity is None:
            self.current_bid_quantity = base_quantity
        
        # Update price based on action
        if action_idx == 0:  # Increase price
            self.current_bid_price += tick_size
        elif action_idx == 1:  # Decrease price
            self.current_bid_price -= tick_size
        # Action 2: Maintain price (no change)
        
        # Update quantity based on action
        if action_idx == 3:  # Increase quantity
            self.current_bid_quantity += base_quantity
        elif action_idx == 4:  # Decrease quantity
            self.current_bid_quantity = max(base_quantity, self.current_bid_quantity - base_quantity)
        # Action 5: Maintain quantity (no change)
        
        # Action 6: Aggressively replenish
        if action_idx == 6:
            self.current_bid_quantity = base_quantity * 3  # Triple the base quantity
        
        return {
            'price': self.current_bid_price,
            'quantity': self.current_bid_quantity
        }
    
    def update_pnl(self, trades, initial_price):
        """Update P&L based on executed trades."""
        for trade in trades:
            if trade['maker_id'] == self.agent_id:
                # Calculate P&L as (initial_price - execution_price) * quantity
                # This represents profit from buying below the initial price
                trade_pnl = (initial_price - trade['price']) * trade['quantity']
                self.pnl_history.append(trade_pnl)
                
                # Update inventory and cash
                self.inventory += trade['quantity']
                self.cash -= trade['price'] * trade['quantity']
                
                # Track executed volume
                self.executed_volume += trade['quantity']
    
    def get_total_pnl(self):
        """Get the cumulative P&L."""
        return sum(self.pnl_history)


class TakerAgent:
    """Agent that executes sell orders with different strategies."""
    
    def __init__(self, total_quantity, execution_steps, strategy='passive', randomness=0.2):
        self.total_quantity = total_quantity
        self.execution_steps = execution_steps
        self.strategy = strategy
        self.randomness = randomness
        self.executed_quantity = 0
        self.remaining_steps = execution_steps
        self.pnl_history = []
        self.execution_schedule = self._create_execution_schedule()
    
    def _create_execution_schedule(self):
        """Create a schedule of quantities to execute at each step."""
        if self.strategy == 'aggressive':
            # Front-loaded execution (larger orders first)
            base_schedule = np.linspace(self.total_quantity / self.execution_steps * 2, 
                                        0, 
                                        self.execution_steps)
            # Ensure the total matches the desired quantity
            base_schedule = base_schedule / np.sum(base_schedule) * self.total_quantity
        elif self.strategy == 'passive':
            # Even distribution with slight randomness
            base_schedule = np.ones(self.execution_steps) * (self.total_quantity / self.execution_steps)
        elif self.strategy == 'reactive':
            # Start with passive, will be adjusted during execution
            base_schedule = np.ones(self.execution_steps) * (self.total_quantity / self.execution_steps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Add randomness
        if self.randomness > 0:
            random_factors = 1 + np.random.uniform(-self.randomness, self.randomness, self.execution_steps)
            schedule = base_schedule * random_factors
            # Ensure the total matches the desired quantity
            schedule = schedule / np.sum(schedule) * self.total_quantity
        else:
            schedule = base_schedule
        
        return schedule
    
    def get_next_order_size(self, market_state=None):
        """Get the next order size based on the strategy."""
        if self.remaining_steps <= 0:
            return 0
        
        # Get the base order size from the schedule
        order_size = self.execution_schedule[self.execution_steps - self.remaining_steps]
        
        # Adjust order size for reactive strategy based on market conditions
        if self.strategy == 'reactive' and market_state is not None:
            # If prices are dropping fast, slow down execution
            price_velocity = market_state.get('price_velocity', 0)
            if price_velocity < -0.01:  # Price is dropping
                order_size *= 0.7  # Reduce order size
            elif price_velocity > 0.01:  # Price is rising
                order_size *= 1.3  # Increase order size
        
        self.remaining_steps -= 1
        return int(order_size)
    
    def update_pnl(self, execution_result, initial_price):
        """Update P&L based on execution results."""
        if execution_result['avg_execution_price'] is not None:
            executed_quantity = execution_result['trades'][0]['quantity'] if execution_result['trades'] else 0
            avg_price = execution_result['avg_execution_price']
            
            # For a seller, PnL is the difference between execution price and initial price
            trade_pnl = (avg_price - initial_price) * executed_quantity
            self.pnl_history.append(trade_pnl)
            
            # Update executed quantity
            self.executed_quantity += executed_quantity
    
    def get_total_pnl(self):
        """Get the cumulative P&L."""
        return sum(self.pnl_history)


class MarketSimulation:
    """Simulation environment for HFT market-making."""
    
    def __init__(self, initial_price=100.0, tick_size=0.01, max_steps=100):
        self.order_book = OrderBook(initial_price, tick_size)
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.max_steps = max_steps
        self.current_step = 0
        self.makers = {}
        self.taker = None
        
        # For tracking simulation results
        self.price_history = [initial_price]
        self.volume_history = []
        self.maker_pnls = {}
        self.taker_pnl = []
    
    def add_maker(self, maker_id, maker_agent):
        """Add a market maker to the simulation."""
        self.makers[maker_id] = maker_agent
        self.maker_pnls[maker_id] = []
    
    def set_taker(self, taker_agent):
        """Set the taker agent for the simulation."""
        self.taker = taker_agent
    
    def step(self):
        """Execute one step of the simulation."""
        if self.current_step >= self.max_steps:
            return True  # Simulation is done
        
        # Get the current market state
        market_state = self.order_book.get_market_state()
        market_state['time_remaining'] = self.max_steps - self.current_step
        market_state['price_velocity'] = self.order_book.get_recent_trading_velocity()
        market_state['volume'] = self.order_book.get_recent_trading_volume()
        
        # Makers place orders
        for maker_id, maker in self.makers.items():
            action_idx = maker.choose_action(market_state)
            order_params = maker.interpret_action(action_idx, self.order_book.current_price, self.tick_size)
            
            # Place the order
            self.order_book.add_bid(maker_id, order_params['price'], order_params['quantity'])
        
        # Taker executes orders
        if self.taker is not None:
            order_size = self.taker.get_next_order_size(market_state)
            
            if order_size > 0:
                execution_result = self.order_book.execute_market_sell(order_size)
                
                # Update taker PnL
                self.taker.update_pnl(execution_result, self.initial_price)
                self.taker_pnl.append(self.taker.get_total_pnl())
                
                # Update maker PnLs
                for maker_id, maker in self.makers.items():
                    maker_trades = [trade for trade in execution_result['trades'] if trade['maker_id'] == maker_id]
                    maker.update_pnl(maker_trades, self.initial_price)
                    self.maker_pnls[maker_id].append(maker.get_total_pnl())
                
                # Track volume
                executed_volume = sum(trade['quantity'] for trade in execution_result['trades'])
                self.volume_history.append(executed_volume)
            else:
                self.volume_history.append(0)
        
        # Track price
        self.price_history.append(self.order_book.current_price)
        
        # Update learning for makers
        next_market_state = self.order_book.get_market_state()
        next_market_state['time_remaining'] = self.max_steps - self.current_step - 1
        next_market_state['price_velocity'] = self.order_book.get_recent_trading_velocity()
        next_market_state['volume'] = self.order_book.get_recent_trading_volume()
        
        for maker_id, maker in self.makers.items():
            # Calculate reward based on recent PnL change
            if len(self.maker_pnls[maker_id]) >= 2:
                reward = self.maker_pnls[maker_id][-1] - self.maker_pnls[maker_id][-2]
            else:
                reward = 0
            
            # Update Q-table
            maker.update_q_table(
                market_state, 
                maker.choose_action(market_state), 
                reward, 
                next_market_state, 
                self.current_step == self.max_steps - 1
            )
            
            # Decay exploration rate
            maker.decay_exploration()
        
        self.current_step += 1
        return self.current_step >= self.max_steps
    
    def run_simulation(self):
        """Run the complete simulation."""
        done = False
        
        while not done:
            done = self.step()
        
        return self.get_results()
    
    def get_results(self):
        """Get the results of the simulation."""
        results = {
            'price_history': self.price_history,
            'volume_history': self.volume_history,
            'maker_pnls': self.maker_pnls,
            'taker_pnl': self.taker_pnl,
            'final_maker_pnls': {maker_id: pnl[-1] if pnl else 0 for maker_id, pnl in self.maker_pnls.items()},
            'maker_volumes': {maker_id: maker.executed_volume for maker_id, maker in self.makers.items()},
            'total_volume': sum(self.volume_history)
        }
        
        # Calculate market share for each maker
        total_maker_volume = sum(results['maker_volumes'].values())
        if total_maker_volume > 0:
            results['market_shares'] = {maker_id: volume / total_maker_volume for maker_id, volume in results['maker_volumes'].items()}
        else:
            results['market_shares'] = {maker_id: 0 for maker_id in self.makers.keys()}
        
        return results


def run_experiment(num_makers=2, taker_strategy='passive', total_quantity=8000, 
                   num_episodes=100, max_steps=60, initial_price=100.0):
    """Run an experiment with the specified parameters."""
    
    # Define action space for makers
    action_space = list(range(7))  # 7 possible actions as defined in interpret_action method
    
    # Keep track of results across episodes
    all_results = []
    
    for episode in tqdm(range(num_episodes), desc=f"Running experiment with {num_makers} makers"):
        # Create the simulation environment
        sim = MarketSimulation(initial_price=initial_price, max_steps=max_steps)
        
        # Create and add makers
        for i in range(num_makers):
            maker = QLearningAgent(f"maker_{i}", action_space, 
                                  learning_rate=0.1, 
                                  exploration_rate=max(0.1, 1.0 - episode / (num_episodes * 0.7)))
            sim.add_maker(f"maker_{i}", maker)
        
        # Create and add taker
        taker = TakerAgent(total_quantity=total_quantity, 
                          execution_steps=max_steps, 
                          strategy=taker_strategy)
        sim.set_taker(taker)
        
        # Run the simulation
        results = sim.run_simulation()
        all_results.append(results)
    
    return all_results


def visualize_results(all_results, num_makers, taker_strategy):
    """Visualize the results of the experiments."""
    
    # Calculate average metrics across all episodes
    avg_price_history = np.mean([r['price_history'] for r in all_results], axis=0)
    avg_volume_history = np.mean([r['volume_history'] for r in all_results], axis=0)
    
    avg_maker_pnls = {}
    for maker_id in all_results[0]['maker_pnls'].keys():
        avg_maker_pnls[maker_id] = np.mean([r['maker_pnls'][maker_id][-1] if r['maker_pnls'][maker_id] else 0 for r in all_results])
    
    avg_market_shares = {}
    for maker_id in all_results[0]['market_shares'].keys():
        avg_market_shares[maker_id] = np.mean([r['market_shares'][maker_id] for r in all_results])
    
    # Plot price and volume history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(avg_price_history)
    plt.title('Average Price History')
    plt.xlabel('Step')
    plt.ylabel('Price')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_volume_history)
    plt.title('Average Volume History')
    plt.xlabel('Step')
    plt.ylabel('Volume')
    
    # Plot maker PnLs
    plt.subplot(2, 2, 3)
    plt.bar(avg_maker_pnls.keys(), avg_maker_pnls.values())
    plt.title('Average Maker PnLs')
    plt.ylabel('PnL')
    
    # Plot market shares
    plt.subplot(2, 2, 4)
    plt.bar(avg_market_shares.keys(), avg_market_shares.values())
    plt.title('Average Market Shares')
    plt.ylabel('Market Share')
    
    plt.tight_layout()
    plt.suptitle(f'{num_makers} Makers vs {taker_strategy.capitalize()} Taker', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    plt.show()


def run_multiple_experiments():
    """Run experiments with different numbers of makers and taker strategies."""
    
    # Define experiment configurations
    maker_counts = [1, 2, 3]
    taker_strategies = ['passive', 'aggressive', 'reactive']
    
    # Store results for all configurations
    all_experiment_results = {}
    
    for num_makers in maker_counts:
        for strategy in taker_strategies:
            print(f"\nRunning experiment with {num_makers} makers and {strategy} taker strategy")
            
            results = run_experiment(
                num_makers=num_makers,
                taker_strategy=strategy,
                total_quantity=8000,
                num_episodes=50,  # Reduced for demonstration
                max_steps=60,
                initial_price=100.0
            )
            
            key = f"{num_makers}_makers_{strategy}_taker"
            all_experiment_results[key] = results
            
            # Visualize results for this configuration
            visualize_results(results, num_makers, strategy)
    
    # Compare performance across different configurations
    compare_configurations(all_experiment_results)


def compare_configurations(all_experiment_results):
    """Compare performance metrics across different configurations."""
    
    # Extract key metrics for comparison
    config_metrics = {}
    
    for config, results in all_experiment_results.items():
        # Average final PnL for each maker
        avg_maker_pnls = {}
        for maker_id in results[0]['maker_pnls'].keys():
            avg_maker_pnls[maker_id] = np.mean([r['maker_pnls'][maker_id][-1] if r['maker_pnls'][maker_id] else 0 for r in results])
        
        # Average PnL per maker
        avg_pnl_per_maker = np.mean(list(avg_maker_pnls.values()))
        
        # Average taker PnL
        avg_taker_pnl = np.mean([r['taker_pnl'][-1] if r['taker_pnl'] else 0 for r in results])
        
        # Average price impact (final price vs initial price)
        avg_price_impact = np.mean([r['price_history'][-1] - r['price_history'][0] for r in results])
        
        config_metrics[config] = {
            'avg_pnl_per_maker': avg_pnl_per_maker,
            'avg_taker_pnl': avg_taker_pnl,
            'avg_price_impact': avg_price_impact
        }
    
    # Plot comparison of maker PnLs across configurations
    plt.figure(figsize=(12, 6))
    
    configs = list(config_metrics.keys())
    maker_pnls = [config_metrics[config]['avg_pnl_per_maker'] for config in configs]
    
    plt.subplot(1, 2, 1)
    plt.bar(configs, maker_pnls)
    plt.title('Average PnL per Maker Across Configurations')
    plt.xlabel('Configuration')
    plt.ylabel('PnL')
    plt.xticks(rotation=45)
    
    # Plot comparison of price impact across configurations
    price_impacts = [config_metrics[config]['avg_price_impact'] for config in configs]
    
    plt.subplot(1, 2, 2)
    plt.bar(configs, price_impacts)
    plt.title('Average Price Impact Across Configurations')
    plt.xlabel('Configuration')
    plt.ylabel('Price Impact')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def analyze_maker_strategies(all_experiment_results):
    """Analyze the effectiveness of different maker strategies."""
    
    # For each configuration, analyze which actions were most frequently taken
    for config, results in all_experiment_results.items():
        if len(results) == 0:
            continue
        
        # We need to extract the policies from the Q-tables of the makers
        # For simplicity, let's just analyze the last episode
        last_result = results[-1]
        
        print(f"\nAnalysis for {config}:")
        
        for maker_id, maker in last_result['makers'].items():
            # Extract the policy from the Q-table
            policy = {}
            for state_key, q_values in maker.q_table.items():
                policy[state_key] = np.argmax(q_values)
            
            # Count action frequencies
            action_counts = np.zeros(7)
            for action in policy.values():
                action_
