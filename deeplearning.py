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
    lidar_range: int = 3  # Simplified Scan  3x3x3
    max_episode_steps: int = 600
    timeout_seconds: float = 30.0

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    entropy_beta: float = 0.02

    # Training settings
    batch_size: int = 128
    update_every: int = 2048
    ppo_epochs: int = 4

    # Network architecture
    hidden_size: int = 128


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
        print(f"[CSV] Log created : {self.filepath}")

    def log_episode(self, episode_data: dict):
        row = [episode_data.get(h, '') for h in self.header]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


# =====================================
# PPO Model
# =====================================
class SimplePPONetwork(nn.Module):
    """Réseau simple mais efficace pour PPO"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        # Shared network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Actor and Critic heads
        self.actor_mean = nn.Linear(hidden_size, 4)  # yaw, pitch, forward, sprint
        self.actor_log_std = nn.Parameter(torch.zeros(4) - 0.5)  # Commence avec std ~0.6
        self.critic = nn.Linear(hidden_size, 1)

        # Initialisation des poids pour stabilité
        self._init_weights()

    def _init_weights(self):
        """Initialisation stable des poids"""
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        # Actor head with a smaller gain for smoother initial actions
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Critic head with standard initialization
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_log_std.clamp(-2, 1))
        value = self.critic(x)

        return action_mean, action_std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


# =====================================
# Minecraft Environment
# =====================================
class SimpleNavigationEnv:
    """A minimalist environment focused on navigation."""

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.reset()

    def reset(self, difficulty="easy"):
        """Resets the environment for a new episode."""
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
            distance = random.uniform(15, 25)
        else:
            distance = random.uniform(25, 40)

        angle = random.uniform(0, 2 * math.pi)
        self.destination = (
            pos[0] + distance * math.cos(angle),
            pos[1],
            pos[2] + distance * math.sin(angle)
        )

        # Mark destination with a glowstone block
        self.marker_pos = (int(self.destination[0]), int(self.destination[1]), int(self.destination[2]))
        ms.execute(f"/setblock {self.marker_pos[0]} {self.marker_pos[1]} {self.marker_pos[2]} minecraft:glowstone")

        # Reset episode state variables
        self.start_distance = self._get_distance()
        self.best_distance = self.start_distance
        self._last_distance = self.start_distance
        self.step_count = 0
        self.start_time = time.time()
        self.stuck_count = 0
        self.reached_5m = False

        # Reset contrôles
        ms.player_press_forward(False)
        ms.player_press_sprint(False)

        self.logger.info(f"Episode reset - Distance: {self.start_distance:.1f}m")

        return self._get_state()

    def _get_distance(self):
        """Calculates the 2D distance to the destination."""
        pos = ms.player_position()
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        return math.sqrt(dx ** 2 + dz ** 2)

    def _get_state(self):
        """Constructs the state vector for the agent."""
        pos = ms.player_position()
        yaw, pitch = ms.player_orientation()

        # Target direction and distance
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        distance = math.sqrt(dx ** 2 + dz ** 2)
        target_yaw = math.degrees(math.atan2(-dx, dz))

        # Normalized yaw error [-1, 1]
        yaw_error = ((target_yaw - yaw + 180) % 360 - 180) / 180.0  # Normalisé [-1, 1]

        # 90-degree LiDAR scan for obstacles
        lidar = self._optimized_lidar_90(pos, yaw)

        # State vector: [distance, yaw_error, pitch, time_elapsed, ...lidar]
        state = [
                    min(distance / 50.0, 1.0),
                    yaw_error,
                    pitch / 90.0,
                    self.step_count / self.config.max_episode_steps,
                ] + lidar

        return np.array(state, dtype=np.float32)

    def _optimized_lidar_90(self, pos, yaw):
        """A fast, minimalist 90° LiDAR using a batch block request."""
        yaw_rad = math.radians(yaw)

        angles = [-30, -15, 0, 15, 30]  # 5 critical rays
        points = []

        # For each ray, check two distances (near and far)
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            ray_yaw = yaw_rad + angle_rad
            for dist in [3, 7]:
                x = int(pos[0] - math.sin(ray_yaw) * dist)
                z = int(pos[2] + math.cos(ray_yaw) * dist)
                y = int(pos[1])
                points.append([x, y, z])

        try:
            blocks = ms.getblocklist(points)
        except:
            return [1.0] * 5  # 5 rayons

        # Process results
        lidar = []
        for i in range(5):  # 5 rays
            block_near = blocks[i * 2]
            block_far = blocks[i * 2 + 1]
            if block_near and block_near != 'minecraft:air':
                lidar.append(0.3)  # Obstacle is very close
            elif block_far and block_far != 'minecraft:air':
                lidar.append(0.7)  # Obstacle is further away
            else:
                lidar.append(1.0)  # Path is clear

        # Expand 5 rays to 9 to match network input size
        expanded = [
            lidar[0], lidar[0],
            lidar[1],
            lidar[2], lidar[2], lidar[2],
            lidar[3],
            lidar[4], lidar[4]
        ]

        return expanded

    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1
        self._apply_action(action)

        # Calculate reward based on progress
        old_distance = self._get_distance() if hasattr(self, '_last_distance') else self.start_distance
        new_distance = self._get_distance()

        # --- Reward Shaping ---
        reward = 0

        # 1. Progress toward target
        progress = old_distance - new_distance
        if progress > 0:
            reward += progress * 15.0  # # Positive reward for getting closer

        else:
            reward += progress * 20.0  # Stronger penalty for moving away

        # 2. Orientation penalty
        pos = ms.player_position()
        yaw, pitch = ms.player_orientation()
        dx = self.destination[0] - pos[0]
        dz = self.destination[2] - pos[2]
        target_yaw = math.degrees(math.atan2(-dx, dz))
        yaw_error = abs(((target_yaw - yaw + 180) % 360 - 180))

        if yaw_error > 90: # Facing the wrong way
            reward -= 0.5
        elif yaw_error > 45:
            reward -= 0.2

        # 3. Time penalty to encourage speed
        reward -= 0.05  # Encourage la vitesse

        # 4. Stuck penalty
        if abs(progress) < 0.01:  # Presque pas bougé
            self.stuck_count = getattr(self, 'stuck_count', 0) + 1
            if self.stuck_count > 10:
                reward -= 1.0  # Forte pénalité si bloqué
        else:
            self.stuck_count = 0

        # 5. BONUS : Close
        if new_distance < 5.0 and not hasattr(self, 'reached_5m'):
            reward += 20.0
            self.reached_5m = True
            self.logger.info(f"Close! Distance: {new_distance:.1f}m")

        # 6. Success bonus
        if new_distance < 2.0:
            reward += 100.0  # Big reward
            self.logger.info(f"SUCCESS! Distance finale: {new_distance:.1f}m")

        # LOG DEBUG
        self.logger.debug(
            f"Step {self.step_count} | Reward: {reward:.2f} | "
            f"Progress: {progress:.3f}m | YawError: {yaw_error:.1f}° | "
            f"Distance: {new_distance:.1f}m | Stuck: {self.stuck_count}"
        )

        # Update distances
        self._last_distance = new_distance
        if new_distance < self.best_distance:
            self.best_distance = new_distance

        # Check if episode is done
        done = False
        if new_distance < 2.0:
            done = True
        elif self.step_count >= self.config.max_episode_steps:
            done = True
            reward -= 10  # Pénalité timeout
        elif time.time() - self.start_time > self.config.timeout_seconds:
            done = True
            reward -= 10

        state = self._get_state()

        info = {
            'distance': new_distance,
            'success': new_distance < 2.0
        }

        return state, reward, done, info

    def _apply_action(self, action):
        """Applies the agent's action to the game."""
        # Actions : [yaw_change, pitch_change, forward_throttle, sprint]
        yaw, pitch = ms.player_orientation()

        # Camera : changements relatifs
        yaw_change = float(action[0]) * 15.0  # Max 15 degrees per step
        pitch_change = float(action[1]) * 5.0

        new_yaw = yaw + yaw_change
        new_pitch = np.clip(pitch + pitch_change, -45, 45)

        ms.player_set_orientation(new_yaw, new_pitch)

        # Movement controls
        throttle = float(action[2])
        sprint = float(action[3]) > 0.5

        # LOG DEBUG
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
# PPO Agent
# =====================================
class SimplePPOAgent:
    """A minimalist but functional PPO agent."""

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else logging.getLogger()

        state_size = 4 + 9
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
        """Selects an action based on the current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Stores a transition in the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """Performs a PPO update with stability protections."""
        if len(self.states) < self.config.batch_size:
            return

        # LOG
        self.logger.info(f"Network update starting with {len(self.states)} samples")

        # Convert buffers to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Calculate returns and advantages
        returns = self._compute_returns()
        advantages = returns - torch.FloatTensor(self.values).to(self.device)

        # IMPORTANT: Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # CLIPPING
        returns = torch.clamp(returns, -100, 100)

        # PPO update loop
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            mean, std, values = self.network(states)
            std = torch.clamp(std, 0.01, 1.0)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -10, 10))

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (Huber loss is more robust than MSE)
            values_clipped = values.squeeze()
            values_clipped = torch.clamp(values_clipped, -100, 100)
            critic_loss = F.huber_loss(values_clipped, returns)  # Plus robuste que MSE

            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            entropy = torch.clamp(entropy, 0, 10)  # Éviter valeurs extrêmes

            loss = actor_loss + 0.5 * critic_loss - self.config.entropy_beta * entropy

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Loss invalide détectée: {loss.item()}, skip update")
                continue

            # Gradient update with clipping
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

            # LOG
            self.logger.debug(
                f"Epoch {epoch + 1}/{self.config.ppo_epochs} | "
                f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | "
                f"Entropy: {entropy:.4f} | Grad Norm: {grad_norm:.3f}"
            )

            # Log si gradients trop grands
            if grad_norm > 5.0:
                self.logger.warning(f"Gradient norm élevé: {grad_norm:.2f}")

            self.optimizer.step()

        # LOG : Fin de mise à jour
        self.logger.info(f"Network update completed - Final grad norm: {grad_norm:.3f}")

        # Clear buffers
        self.clear_buffers()

    def _compute_returns(self):
        """Computes discounted returns."""
        returns = []
        R = 0

        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.config.gamma * R
            returns.insert(0, R)

        return torch.FloatTensor(returns).to(self.device)

    def clear_buffers(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path):
        """Saves the model weights."""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        print(f"[SAVE] Model saved to {path}")

    def load(self, path):
        """Loads model weights from a file."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"[LOAD] Model loaded from {path}")
            return True
        return False


# =====================================
#  Main Training Loop
# =====================================
def train():
    """Main training function."""
    config = Config()
    logger = setup_logger()
    csv_logger = CSVLogger()

    env = SimpleNavigationEnv(config, logger)
    agent = SimplePPOAgent(config, logger)

    agent.load("simple_ppo_model.pth")

    episode_rewards = deque(maxlen=100)
    total_steps = 0

    for episode in range(1000):
        # Progressive difficulty (curriculum learning)
        if episode < 200:
            difficulty = "easy"
        elif episode < 500:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Reset
        state = env.reset(difficulty)
        episode_reward = 0
        steps = 0

        # Episode
        while True:
            # Action
            action, log_prob, value = agent.select_action(state)

            # Step
            next_state, reward, done, info = env.step(action)

            # Store
            agent.store_transition(state, action, reward, log_prob, value, done)

            # Update counters
            episode_reward += reward
            steps += 1
            total_steps += 1
            state = next_state

            # Update network
            if total_steps % config.update_every == 0:
                agent.update()
                print(f"[UPDATE] Update networks after {total_steps} steps")

            if done:
                break

        # Log episode
        episode_rewards.append(episode_reward)

        # CSV log
        csv_logger.log_episode({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'episode': episode,
            'total_reward': episode_reward,
            'steps': steps,
            'success': info['success'],
            'final_distance': info['distance'],
            'start_distance': env.start_distance
        })

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"\n[Episode {episode}]")
            print(f"  Difficulty: {difficulty}")
            print(f"  Average Reward (Last 100): {avg_reward:.2f}")
            print(f"  Last episode: reward={episode_reward:.2f}, steps={steps}, success={info['success']}")

        # Save
        if episode % 50 == 0 and episode > 0:
            agent.save("simple_ppo_model.pth")

    # Cleanup
    csv_logger.close()
    print("\n[TERMINÉ] Training complete!")


# =====================================
# Entry Point
# =====================================
if __name__ == "__main__":
    print("=" * 50)
    print("   SIMPLE PPO - MINECRAFT NAVIGATION")
    print("=" * 50)
    print("\nStarting training...")

    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()