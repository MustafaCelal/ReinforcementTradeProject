from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in Tensorboard.
    Tracks equity, win rate, trades, and reward components.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Access the environment via self.training_env
        # DummyVecEnv wraps the original env
        
        # We need to access the info dict to get custom metrics
        # The info dict is returned by env.step(), which SB3 stores in self.locals['infos']
        infos = self.locals.get("infos", [{}])
        
        for info in infos:
            if "episode" in info:
                # Standard SB3 episode logging (reward, length) is handled automatically
                pass
            
            # Extract custom metrics from info if available
            equity = info.get("equity_usd")
            position = info.get("position")
            
            if equity is not None:
                self.logger.record("trading/equity", equity)
                
            if position is not None:
                self.logger.record("trading/position", position)
                
            # Log events (trades)
            events = info.get("events", [])
            for event in events:
                if event.get("event") == "CLOSE":
                    pnl = event.get("net_pips", 0.0)
                    self.logger.record("trading/realized_pnl_pips", pnl)
                    self.logger.record("trading/win", 1 if pnl > 0 else 0)
                    
                    # Log R:R if available
                    rr = event.get("rr_ratio")
                    if rr:
                        self.logger.record("trading/rr_ratio_actual", rr)

        return True
        
    def _on_rollout_end(self) -> None:
        # Access the environment's internal state if needed
        # This is called after n_steps
        pass
