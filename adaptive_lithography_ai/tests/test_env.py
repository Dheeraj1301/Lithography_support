def test_env_reset_and_step():
    from src.envs.litho_env import LithoEnv
    env = LithoEnv()
    state = env.reset()
    assert len(state) == 5
    new_state, reward, done, _ = env.step([0.1, 0.2])
    assert len(new_state) == 5
