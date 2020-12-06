# RL_Algorithms
A collection of RL algorithms within a modular code structure for easy additions of new algorithms.

# Currently Implemented
The following algorithms have been implemented and have solved the OpenAI gym CartPole environments. The reward plots can be seen in ./plots.
- DDQN (Double Deep Q-Network) Solved at episode 516
    * model = DDQN
    * agent = DDQNAgent
    * trainer = DDQNTrainer
    * hidden = [128,128]
    * buffer_size = 1000
- A2C (Advantage Actor-Critic) Solved at episode 550
    * model = ActorCritic
    * agent = A2CAgent
    * trainer = A2CTrainer
    * hidden = [128]
    * buffer_size = 300

# To Run
To train a model, change hyperparameters within the ./utils.py file and run ./main.py.
