# 🎮 Minecraft PPO Navigation Agent

A reinforcement learning agent that learns to navigate in Minecraft using Proximal Policy Optimization (PPO) and the [Minescript](https://github.com/minescript/minescript) mod.

## 🚀 Overview

This project implements a PPO-based agent that learns to:
- Navigate to marked destinations (glowstone blocks)
- Control camera movement smoothly
- Avoid obstacles using a simple LiDAR system
- Sprint when appropriate

The agent uses a simplified but effective approach focused on practical navigation rather than complex architectures.

## 📋 Features

- **Simple PPO Implementation**: Clean, understandable code with proper gradient clipping
- **90° LiDAR Vision**: 9-ray system matching the player's field of view
- **Progressive Difficulty**: Curriculum learning from 5m to 40m distances
- **Robust Training**: Includes normalizations, clipping, and NaN/Inf detection
- **Comprehensive Logging**: CSV metrics and debug logs for analysis
- **Updates**: Network updates every 2048 steps for responsive learning

## 🛠️ Requirements

```bash
# Minecraft with Minescript 4.0  mod installed
# Python 3.8+
pip install torch numpy 
```

## 📁 Project Structure

```
reinforcement.py  # Main training script
data/
  └── training_log_*.csv      # Episode metrics
agent_debug.log               # Detailed debug information
simple_ppo_model.pth         # Saved model weights
```

## 🎯 How It Works

### State Space (13 dimensions)
- Distance to target (normalized)
- Yaw error to target
- Current pitch
- Time elapsed ratio
- 9 LiDAR readings (obstacle distances)

### Action Space (4 continuous)
- Yaw change (-15° to +15° per step)
- Pitch change (-5° to +5° per step)
- Forward throttle (-1 to 1)
- Sprint (0 or 1)

### Reward System
```python
# Positive rewards
+15/meter for approaching target
+20 bonus at 5m distance
+100 for reaching destination

# Penalties
-20/meter for moving away
-0.5 for wrong orientation (>90°)
-1.0 for being stuck
-0.05 per step (encourages speed)
```

## 🚦 Getting Started

1. **Launch Minecraft** with Minescript mod
2. **Enter a flat world** (recommended for initial training)
3. **Run the training**:
```bash
\reinforcement
```

4. **Monitor progress** via console output and CSV logs

## 📊 Training Phases

| Phase | Episodes | Distance | Focus |
|-------|----------|----------|-------|
| Easy | 0-500    | 5-10m    | Learn basic navigation |
| Medium | 500-1000 | 15-25m   | Improve efficiency |
| Hard | 1000+    | 25-40m   | Master long distances |

## 🔧 Configuration

Key parameters in `Config` class:
```python
lr: float = 3e-4           # Learning rate
gamma: float = 0.99        # Discount factor
update_every: int = 2048    # Steps between updates
hidden_size: int = 128     # Network size
```

Expected training time: ~2-4 hours for basic navigation

## 🤝 Contributing

This is an active learning project! Contributions welcome:
- Performance optimizations
- Better reward shaping
- Advanced LiDAR systems
- Terrain handling improvements

### Ideas for Improvement
- [ ] Add jumping over obstacles
- [ ] Implement pathfinding memory
- [ ] Handle water/lava
- [ ] Multi-target navigation
- [ ] Transfer learning to complex terrains

## 📝 Logging & Analysis

**CSV Logs** include:
- Episode number and reward
- Success rate
- Final distance
- Step count

**Debug Log** tracks:
- Action decisions
- Reward components
- Network updates
- Gradient norms

## 🔍 Technical Details

The implementation uses:
- **Orthogonal initialization** for stable training
- **Huber loss** for critic (robust to outliers)
- **Gradient clipping** at 1.0
- **Advantage normalization**
- **Batch processing** with `getblocklist()` for efficiency

## 🐛 Known Issues

- LiDAR can be slow on complex terrain


## 💬 Community

Join the discussion:
- Open an issue for bugs/suggestions
- Share your training results!

## 📄 License

MIT License - Feel free to use and modify!

---

**Note**: This is a learning project. The agent needs several hours of training to show good results. Patience and experimentation are key! 🎯

*If you find this helpful, consider leaving a ⭐!*