import gym
from gym import spaces
import numpy as np

class DiabetesEnv(gym.Env):
    """
    Custom Environment for simulating diabetic patient treatment.
    The agent decides how to adjust treatment based on glucose level, diet, and activity.
    """
    def __init__(self):
        super(DiabetesEnv, self).__init__()

        # Action space: 0 = more insulin, 1 = less insulin, 2 = more exercise, 3 = better diet
        self.action_space = spaces.Discrete(4)

        # Observation: [blood_glucose, carb_intake, activity_level]
        self.observation_space = spaces.Box(
            low=np.array([50, 0, 0]), 
            high=np.array([400, 200, 5]), 
            dtype=np.float32
        )

        self.state = None
        self.day = 0
        self.max_days = 30

    def reset(self):
        self.state = np.array([180.0, 70.0, 2.0], dtype=np.float32)  # Starting glucose, carbs, activity
        self.day = 0
        return self.state

    def step(self, action):
        glucose, carbs, activity = self.state

        # Simulate effects of actions
        if action == 0:  # more insulin
            glucose -= 15
        elif action == 1:  # less insulin
            glucose += 10
        elif action == 2:  # more exercise
            activity = min(activity + 1, 5)
            glucose -= 5
        elif action == 3:  # better diet
            carbs = max(carbs - 10, 0)
            glucose -= 3

        # Add some randomness to simulate variability
        glucose += np.random.normal(0, 5)

        # Update state
        self.state = np.array([glucose, carbs, activity], dtype=np.float32)
        self.day += 1

        # Reward logic
        if 80 <= glucose <= 130:
            reward = 1
        else:
            reward = -1

        done = self.day >= self.max_days

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Day {self.day} - Glucose: {self.state[0]:.2f} mg/dL")
