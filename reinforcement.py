# navigation_ppo_simplified.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import minescript as ms
import csv
import os
import time
import random
import logging
from datetime import datetime
from collections import deque
import math


# =====================================
# Configuration
# =====================================
class Config:
    """Simplified project configuration"""
    # Environment settings
    scan_radius: int = 5  # Radius for block scanning
    max_episode_steps: int = 600
    timeout_seconds: float = 30.0

    # PPO hyperparameters - REDUCED learning rate
    lr: float = 3e-5  # Reduced from 3e-4 to prevent explosions
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    entropy_beta: float = 0.01  # Reduced for stability

    # Training settings
    batch_size: int = 128
    update_every: int = 256  # More frequent updates
    ppo_epochs: int = 3  # Reduced epochs

    # Network architecture
    hidden_size: int = 128

    # Gradient clipping - STRICTER
    max_grad_norm: float = 0.5  # Reduced from 1.0


# =====================================
# Logging Setup
# =====================================
def setup_logger():
    """Configures file and console logging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='agent_debug.log',
        filemode='w',
        force=True
    )
    logger = logging.getLogger()

    # Add a console handler for important (INFO level) logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger


class CSVLogger:
    """Logs episode metrics to a CSV file for analysis."""

    def __init__(self):
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join('data', f'training_log_{timestamp}.csv')
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.header = [
            'timestamp', 'episode', 'total_reward', 'steps',
            'success', 'final_distance', 'start_distance'
        ]
        self.writer.writerow(self.header)
        print(f"[CSV] Log created: {self.filepath}")

    def log_episode(self, episode_data: dict):
        row = [episode_data.get(h, '') for h in self.header]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


# =====================================
# Block Cache System
# =====================================
class BlockCache:
    """Caches block data to avoid redundant queries"""

    def __init__(self, ttl=10):
        self.cache = {}
        self.ttl = ttl  # Time to live in steps
        self.current_step = 0

    def get_blocks(self, points):
        """Get blocks with caching"""
        self.current_step += 1

        # Clean expired entries
        if self.current_step % 50 == 0:
            self._clean_expired()

        uncached_points = []
        cached_results = {}

        for point in points:
            key = tuple(point)
            if key in self.cache:
                entry = self.cache[key]
                if self.current_step - entry['step'] < self.ttl:
                    cached_results[key] = entry['block']
                else:
                    uncached_points.append(point)
            else:
                uncached_points.append(point)

        # Fetch uncached blocks
        if uncached_points:
            try:
                new_blocks = ms.getblocklist(uncached_points)
                for i, point in enumerate(uncached_points):
                    key = tuple(point)
                    self.cache[key] = {
                        'block': new_blocks[i],
                        'step': self.current_step
                    }
                    cached_results[key] = new_blocks[i]
            except:
                pass

        # Return blocks in original order
        result = []
        for point in points:
            key = tuple(point)
            result.append(cached_results.get(key, 'minecraft:air'))

        return result

    def _clean_expired(self):
        """Remove expired entries"""
        expired_keys = []
        for key, entry in self.cache.items():
            if self.current_step - entry['step'] >= self.ttl:
                expired_keys.append(key)
        for key in expired_keys:
            del self.cache[key]


# =====================================
# PPO Model with Better Initialization
# =====================================
class SimplePPONetwork(nn.Module):
    """Simple but effective PPO network"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        # Shared network layers with LayerNorm for stability
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Actor and Critic heads
        self.actor_mean = nn.Linear(hidden_size, 4)
        self.actor_log_std = nn.Parameter(torch.zeros(4) - 1.0)  # Lower initial std
        self.critic = nn.Linear(hidden_size, 1)

        # Weight initialization for stability
        self._init_weights()

    def _init_weights(self):
        """Stable weight initialization"""
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        # Very small actor initialization to start with small actions
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.001)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Standard critic initialization
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x):
        # Forward pass with LayerNorm
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_log_std.clamp(-3, 0))  # Stricter std bounds
        value = self.critic(x)

        return action_mean, action_std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        # Ensure action is bounded
        action = torch.tanh(action)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


# =====================================
# Minecraft Environment with Hazard Detection
# =====================================
class SimpleNavigationEnv:
    """Environment with hazard awareness and block caching"""

    # Define hazardous blocks
    HAZARD_BLOCKS = {
        'minecraft:lava': -50.0,
        'minecraft:flowing_lava': -50.0,
        'minecraft:fire': -30.0,
        'minecraft:magma_block': -20.0,
        'minecraft:cactus': -10.0,
        'minecraft:water': -2.0,
        'minecraft:flowing_water': -2.0,
    }

    SOLID_BLOCKS = {
        'minecraft:stone', 'minecraft:dirt', 'minecraft:grass_block',
        'minecraft:cobblestone', 'minecraft:sand', 'minecraft:gravel',
        'minecraft:oak_log', 'minecraft:oak_planks', 'minecraft:bedrock'
    }

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.block_cache = BlockCache(ttl=20)
        self.reset()

    def reset(self, difficulty="easy"):
        """Resets the environment for a new episode."""
        # Clean up previous marker
        if hasattr(self, 'marker_pos'):
            try:
                ms.execute(f"/setblock {self.marker_pos[0]} {self.marker_pos[1]} {self.marker_pos[2]} minecraft:air")
            except:
                pass

        pos = ms.player_position()

        # Set distance based on curriculum difficulty
        if difficulty == "easy":
            distance = random.uniform(5, 10)
        elif difficulty == "medium":
            distance = random.uniform(10, 20)
        else:
            distance = random.uniform(20, 35)

        angle = random.uniform(0, 2 * math.pi)
        self.destination = (
            pos[0] + distance * math.cos(angle),
            pos[1],
            pos[2] + distance * math.sin(angle)
        )

        # Mark destination with glowstone
        self.marker_pos = (int(self.destination[0]), int(self.destination[1]), int(self.destination[2]))
        ms.execute(f"/setblock {self.marker_pos[0]} {self.marker_pos[1]} {self.marker_pos[2]} minecraft:glowstone")

        # Reset episode state
        self.start_distance = self._get_distance()
        self.best_distance = self.start_distance
        self._last_distance = self.start_distance
        self.step_count = 0
        self.start_time = time.time()
        self.stuck_count = 0
        self.reached_5m = False
        self.cumulative_reward = 0

        # Reset controls
        ms.player_press_forward(False)
        ms.player_press_sprint(False)

        self.logger.info(f"Episode reset - Distance: {self.start_distance:.1f}m")

        return self._get_state()

    def _get_distance(self):
        """Calculate 2D distance to destination"""
        pos = ms.player_position()
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        return math.sqrt(dx ** 2 + dz ** 2)

    def _get_state(self):
        """Construct state vector with hazard awareness"""
        pos = ms.player_position()
        yaw, pitch = ms.player_orientation()

        # Target direction and distance
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        distance = math.sqrt(dx ** 2 + dz ** 2)
        target_yaw = math.degrees(math.atan2(-dx, dz))

        # Normalized yaw error
        yaw_error = ((target_yaw - yaw + 180) % 360 - 180) / 180.0

        # Scan surrounding blocks for hazards
        hazard_scan = self._scan_for_hazards(pos, yaw)

        # State vector
        state = [
                    min(distance / 50.0, 1.0),
                    yaw_error,
                    pitch / 90.0,
                    self.step_count / self.config.max_episode_steps,
                ] + hazard_scan

        return np.array(state, dtype=np.float32)

    def _scan_for_hazards(self, pos, yaw):
        """Scan for hazards and obstacles in front"""
        scan_data = []
        yaw_rad = math.radians(yaw)

        # 5 forward rays at different angles
        angles = [-30, -15, 0, 15, 30]

        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            ray_yaw = yaw_rad + angle_rad

            # Check multiple distances
            hazard_level = 0.0
            for dist in [2, 4, 6]:
                x = int(pos[0] - math.sin(ray_yaw) * dist)
                z = int(pos[2] + math.cos(ray_yaw) * dist)

                # Check ground and eye level
                points_to_check = [
                    [x, int(pos[1] - 1), z],  # Ground
                    [x, int(pos[1]), z],  # Feet
                    [x, int(pos[1] + 1), z]  # Head
                ]

                blocks = self.block_cache.get_blocks(points_to_check)

                for block in blocks:
                    if block in self.HAZARD_BLOCKS:
                        hazard_level = min(hazard_level, self.HAZARD_BLOCKS[block] / 50.0)
                    elif block in self.SOLID_BLOCKS:
                        hazard_level = min(hazard_level, -0.1)  # Minor penalty for obstacles

                if hazard_level < 0:
                    break  # Stop checking further if hazard found

            scan_data.append(hazard_level)

        # Expand to 9 values for network compatibility
        expanded = [
            scan_data[0], scan_data[0],
            scan_data[1],
            scan_data[2], scan_data[2], scan_data[2],
            scan_data[3],
            scan_data[4], scan_data[4]
        ]

        return expanded

    def step(self, action):
        """Execute one step with normalized rewards"""
        self.step_count += 1
        self._apply_action(action)

        # Distance calculations
        old_distance = self._last_distance
        new_distance = self._get_distance()

        # NORMALIZED REWARD SYSTEM (-1 to +1 range)
        reward = 0

        # 1. Progress reward (normalized)
        progress = (old_distance - new_distance) / 10.0  # Divide by 10 for normalization
        reward += np.clip(progress * 5.0, -1.0, 1.0)

        # 2. Orientation penalty (small, continuous)
        pos = ms.player_position()
        yaw, _ = ms.player_orientation()
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        target_yaw = math.degrees(math.atan2(-dx, dz))
        yaw_error = abs(((target_yaw - yaw + 180) % 360 - 180))

        reward -= (yaw_error / 180.0) * 0.1  # Small orientation penalty

        # 3. Time penalty (very small)
        reward -= 0.01

        # 4. Stuck penalty
        if abs(progress) < 0.001:
            self.stuck_count += 1
            if self.stuck_count > 20:
                reward -= 0.5
        else:
            self.stuck_count = 0

        # 5. Check for hazards around player
        hazard_penalty = self._check_hazards()
        reward += hazard_penalty

        # 6. Milestone bonus (normalized)
        if new_distance < 5.0 and not self.reached_5m:
            reward += 1.0
            self.reached_5m = True
            self.logger.info(f"Milestone reached! Distance: {new_distance:.1f}m")

        # 7. Success bonus (normalized)
        if new_distance < 2.0:
            reward += 5.0  # Big but normalized reward
            self.logger.info(f"SUCCESS! Final distance: {new_distance:.1f}m")

        # IMPORTANT: Clip total reward to prevent explosions
        reward = np.clip(reward, -2.0, 2.0)

        # Track cumulative reward for stability monitoring
        self.cumulative_reward += reward

        # Debug logging
        self.logger.debug(
            f"Step {self.step_count} | Reward: {reward:.3f} | "
            f"Progress: {progress:.3f} | YawError: {yaw_error:.1f}° | "
            f"Distance: {new_distance:.1f}m | Cumulative: {self.cumulative_reward:.2f}"
        )

        # Update state tracking
        self._last_distance = new_distance
        if new_distance < self.best_distance:
            self.best_distance = new_distance

        # Check termination
        done = False
        if new_distance < 2.0:
            done = True
        elif self.step_count >= self.config.max_episode_steps:
            done = True
        elif time.time() - self.start_time > self.config.timeout_seconds:
            done = True

        state = self._get_state()
        info = {
            'distance': new_distance,
            'success': new_distance < 2.0
        }

        return state, reward, done, info

    def _check_hazards(self):
        """Check for hazards at player position"""
        pos = ms.player_position()
        points = [
            [int(pos[0]), int(pos[1] - 1), int(pos[2])],  # Below feet
            [int(pos[0]), int(pos[1]), int(pos[2])]  # At feet
        ]

        blocks = self.block_cache.get_blocks(points)

        penalty = 0
        for block in blocks:
            if block in self.HAZARD_BLOCKS:
                penalty = self.HAZARD_BLOCKS[block] / 50.0  # Normalized penalty
                self.logger.debug(f"Hazard detected: {block} (penalty: {penalty:.2f})")
                break

        return penalty

    def _apply_action(self, action):
        """Apply bounded actions to the game"""
        yaw, pitch = ms.player_orientation()

        # Actions are already in [-1, 1] from tanh
        yaw_change = float(action[0]) * 10.0  # Max 10° per step (reduced)
        pitch_change = float(action[1]) * 3.0  # Max 3° per step (reduced)

        new_yaw = yaw + yaw_change
        new_pitch = np.clip(pitch + pitch_change, -45, 45)

        ms.player_set_orientation(new_yaw, new_pitch)

        # Movement controls
        throttle = float(action[2])
        sprint = float(action[3]) > 0

        self.logger.debug(
            f"Actions: Yaw={yaw_change:.1f}° Pitch={pitch_change:.1f}° "
            f"Throttle={throttle:.2f} Sprint={sprint}"
        )

        if throttle > 0.1:
            ms.player_press_forward(True)
            ms.player_press_backward(False)
            ms.player_press_sprint(sprint)
        elif throttle < -0.1:
            ms.player_press_forward(False)
            ms.player_press_backward(True)
            ms.player_press_sprint(False)
        else:
            ms.player_press_forward(False)
            ms.player_press_backward(False)
            ms.player_press_sprint(False)


# =====================================
# Improved PPO Agent
# =====================================
class SimplePPOAgent:
    """PPO agent with improved stability"""

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else logging.getLogger()

        state_size = 4 + 9  # 4 base features + 9 hazard scan values
        self.network = SimplePPONetwork(state_size, config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """Select action with exploration noise decay"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition with reward clipping"""
        # Clip reward before storing to prevent value function explosion
        reward = np.clip(reward, -2.0, 2.0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """PPO update with enhanced stability measures"""
        if len(self.states) < self.config.batch_size:
            return

        self.logger.info(f"Network update starting with {len(self.states)} samples")

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute returns with reward normalization
        returns = self._compute_returns()

        # CRITICAL: Normalize returns for stability
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        returns = (returns - returns_mean) / returns_std

        # Compute advantages
        values_tensor = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO optimization loop
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            mean, std, values = self.network(states)

            # Ensure std stays in reasonable range
            std = torch.clamp(std, 0.01, 0.5)

            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # PPO ratio with log space for numerical stability
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(torch.clamp(log_ratio, -5, 5))  # Tighter bounds

            # Surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,
                                1 - self.config.clip_epsilon,
                                1 + self.config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss with clipping
            values_pred = values.squeeze()
            values_clipped = values_tensor + torch.clamp(
                values_pred - values_tensor,
                -self.config.clip_epsilon,
                self.config.clip_epsilon
            )
            value_loss_unclipped = F.mse_loss(values_pred, returns)
            value_loss_clipped = F.mse_loss(values_clipped, returns)
            critic_loss = torch.max(value_loss_unclipped, value_loss_clipped)

            # Entropy for exploration
            entropy = dist.entropy().mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - self.config.entropy_beta * entropy

            # Skip update if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Invalid loss detected: {loss.item()}, skipping update")
                continue

            # Gradient update with strict clipping
            self.optimizer.zero_grad()
            loss.backward()

            # CRITICAL: Strict gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm
            )

            # Log gradient information
            self.logger.debug(
                f"Epoch {epoch + 1}/{self.config.ppo_epochs} | "
                f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | "
                f"Entropy: {entropy:.4f} | Grad Norm: {grad_norm:.3f}"
            )

            # Warning for high gradients (but they're already clipped)
            if grad_norm > self.config.max_grad_norm * 0.9:
                self.logger.warning(f"Gradient norm near limit: {grad_norm:.2f}")

            self.optimizer.step()

        self.logger.info("Network update completed")
        self.clear_buffers()

    def _compute_returns(self):
        """Compute discounted returns with reward normalization"""
        returns = []
        R = 0

        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.config.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Clip returns to prevent extreme values
        returns_tensor = torch.clamp(returns_tensor, -10, 10)

        return returns_tensor

    def clear_buffers(self):
        """Clear experience buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        print(f"[SAVE] Model saved to {path}")

    def load(self, path):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"[LOAD] Model loaded from {path}")
            return True
        return False


# =====================================
# Main Training Loop
# =====================================
best_avg_reward = -float('inf')
def train():
    """Main training function with improved stability"""
    global best_avg_reward
    config = Config()
    logger = setup_logger()
    csv_logger = CSVLogger()

    env = SimpleNavigationEnv(config, logger)
    agent = SimplePPOAgent(config, logger)

    # Load existing model if available
    agent.load("simple_ppo_model.pth")

    episode_rewards = deque(maxlen=100)
    total_steps = 0

    for episode in range(1000):
        # Curriculum learning
        if episode < 200:
            difficulty = "easy"
        elif episode < 500:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Reset environment
        state = env.reset(difficulty)
        episode_reward = 0
        steps = 0

        # Run episode
        while True:
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)

            # Update counters
            episode_reward += reward
            steps += 1
            total_steps += 1
            state = next_state

            # Update network
            if total_steps % config.update_every == 0:
                agent.update()
                logger.info(f"[UPDATE] Network updated after {total_steps} steps")

            if done:
                break

        # Log episode
        episode_rewards.append(episode_reward)

        # CSV logging
        csv_logger.log_episode({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'episode': episode,
            'total_reward': episode_reward,
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'start_distance': env.start_distance
        })

        episode_rewards.append(episode_reward)
        # Console output
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            success_rate = sum(1 for r in episode_rewards if r > 5) / len(episode_rewards) * 100
            print(f"\n[Episode {episode}]")
            print(f"  Difficulty: {difficulty}")
            print(f"  Avg Reward (100 eps): {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Last: reward={episode_reward:.2f}, steps={steps}, success={info['success']}")
            if avg_reward > best_avg_reward and len(episode_rewards) >= 100:  # On attend d'avoir assez de données
                best_avg_reward = avg_reward
                agent.save("best_model.pth")  # Sauvegarde dans un fichier séparé
                print(f"[SAVE] New best model saved with avg reward: {avg_reward:.2f}")

        # Save model
        if episode % 50 == 0 and episode > 0:
            agent.save("checkpoint_model.pth")

    # Cleanup
    csv_logger.close()
    print("\n[COMPLETE] Training finished!")


# =====================================
# Entry Point
# =====================================
if __name__ == "__main__":
    print("=" * 50)
    print("   PPO MINECRAFT NAVIGATION - v2.0")
    print("   With Hazard Detection & Stability Fixes")
    print("=" * 50)
    print("\nStarting training...")

    try:
        train()
    except KeyboardInterrupt:
        print("\n[STOPPED] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()