# Monopoly RL Advanced

A compact, educational Monopoly simulator with **Deep Q-Learning (DQN)**, richer game rules, multi-item trading, scripted opponents, and detailed logging. This project is designed for reinforcement learning research and experimentation, not for full rules-accurate gameplay.

---

## Features

- **Fuller Monopoly Rules**:
    - Chance & Community Chest events
    - Jail (go to jail, get out by rolling doubles or paying)
    - Utilities & Railroads
    - Auctions when a player declines to buy
    - Simplified mortgaging (auto-mortgage lowest-value properties if cash < 0)
- **Trading**:
    - Multi-item (k-for-m) trades with optional cash sweetener
    - $100 premium for trades that complete a set for the receiver
    - Stochastic "lean to accept" behavior near fair trades
    - Safety checks for valid trades
- **Agent**:
    - DQN (MLP) with target network
    - Epsilon-greedy exploration with decay
    - Experience replay buffer
- **Opponents**:
    - Toggle between scripted bots (aggressive buyer, conservative, builder) or self-play pool
- **Evaluation & Logging**:
    - Tracks landing distribution, win rates, average wealth
    - CSV logs and weights checkpoints
- **CLI Flags**:
    - Control episodes, evaluation frequency, opponent type, and save directory

---

## Quick Start

### Requirements

- Python 3.7+
- No external dependencies (uses only `numpy`, `argparse`, `csv`, etc.)

### Usage

```bash
python monopoly_rl_advanced.py --episodes 5000 --eval-every 250 --opponents scripted --save-dir ./mono_adv
```

**Arguments:**
- `--episodes`: Number of training episodes (default: 1000)
- `--eval-every`: Evaluation interval (default: 100)
- `--opponents`: Opponent type (`scripted` or `selfplay`)
- `--save-dir`: Directory for logs and weights

---

## How It Works

### Game Mechanics

- **Board**: 40 tiles, including properties, railroads, utilities, taxes, jail, chance/chest, free parking, and go-to-jail.
- **Players**: 4 per game, each with cash, properties, houses, and mortgage status.
- **Actions**: Skip, Buy, Build, Offer Trade, Pay to Leave Jail.
- **Trading**: Agents can propose multi-property trades with cash, evaluated stochastically for fairness.
- **Mortgaging**: If cash drops below zero, properties are auto-mortgaged until solvency or bankruptcy.

### RL Agent

- **Observation**: Encodes ownership, houses, mortgages, position, cash, jail status, and completed sets.
- **DQN**: Simple MLP (implemented in numpy), trained via experience replay and target network updates.
- **Exploration**: Epsilon-greedy, decaying over time.

### Opponents

- **Scripted Bots**: Aggressive, conservative, and builder styles.
- **Self-Play**: All agents use the same DQN policy.

### Logging & Evaluation

- **CSV Log**: Records episode, winner, moves, agent epsilon, average wealth, agent wealth, evaluation win rate, and average moves.
- **Weights**: Saved every evaluation interval and at the end.

---

## Output Files

- `training_log.csv`: Episode-by-episode stats.
- `weights_ep{N}.json`: DQN weights at evaluation checkpoints.
- `weights_final.json`: Final trained weights.

---

## Limitations

- **Not a full Monopoly engine**: Some rules are simplified for RL speed and clarity.
- **No GUI**: CLI only.
- **No external RL libraries**: All neural network code is pure numpy.

---

## Extending

- Add more sophisticated agents or opponent strategies.
- Tune DQN architecture or hyperparameters.
- Integrate with external RL libraries for advanced experimentation.

---

## License

This project is for educational and research use. See [LICENSE](LICENSE) for details.

---

## Credits

Vibecoded

---

## Contact

For questions or suggestions, open an issue or contact chatgpt@asu.edu (Yes this email is valid).
