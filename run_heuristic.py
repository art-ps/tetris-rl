from tetris_rl.agents import HeuristicAgent
from tetris_rl.env import TetrisEnv


env = TetrisEnv(seed=42)
agent = HeuristicAgent()

done = False
total_reward = 0.0

while not done:
    valid_actions = env.get_valid_actions()
    action_index = agent.select_action(valid_actions)
    state, reward, done, info = env.step(action_index)
    total_reward += reward

print(env.render_text())
print(f"Reward: {total_reward:.2f}")
print(f"Score: {info['score']}")
print(f"Lines: {info['lines_cleared']}")
print(f"Pieces: {info['pieces_placed']}")
print(f"Max height: {info['max_height']}")
print(f"Holes: {info['holes']}")
