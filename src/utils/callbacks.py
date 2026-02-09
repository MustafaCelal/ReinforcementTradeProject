from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import os

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


class BestModelEvalCallback(BaseCallback):
    """
    Best Model Evaluation Callback
    
    Düzenli olarak modeli değerlendirir ve en yüksek final equity'ye sahip modeli kaydeder.
    Bu, eğitim sırasında tüm checkpointleri manuel olarak test etmeden, 
    en iyi performans gösteren modeli otomatik yakalamayı sağlar.
    
    Özellikleri:
    - eval_freq timestep'te bir değerlendirme yapar
    - Final equity'yi takip eder
    - En yüksek equity'ye sahip modeli best_model_save_path'e kaydeder
    - TensorBoard'a best equity'yi log eder
    
    Parametreler:
        eval_env: Değerlendirme ortamı (DummyVecEnv)
        eval_freq: Kaç timestep'te bir değerlendirme yapılacağı
        best_model_save_path: En iyi modelin kaydedileceği path
        verbose: Verbose level
    """
    
    def __init__(self, eval_env, eval_freq=50_000, best_model_save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.best_equity = -np.inf
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            # Perform evaluation (supports both PPO and RecurrentPPO)
            obs = self.eval_env.reset()
            equity_curve = []
            
            # For RecurrentPPO: LSTM states management
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while True:
                # Try RecurrentPPO predict (with state parameter)
                # If it fails, fall back to regular PPO predict
                try:
                    action, lstm_states = self.model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True
                    )
                except TypeError:
                    # Fallback untuk regular PPO (state parameter yok)
                    action, _ = self.model.predict(obs, deterministic=True)
                
                step_out = self.eval_env.step(action)
                
                if len(step_out) == 4:
                    obs, rewards, dones, infos = step_out
                    done = bool(dones[0])
                else:
                    obs, rewards, terminated, truncated, infos = step_out
                    done = bool(terminated[0] or truncated[0])
                
                # Update episode_starts for RecurrentPPO
                if len(step_out) == 4:
                    episode_starts = dones
                else:
                    episode_starts = np.array([bool(terminated[0] or truncated[0])])
                
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                eq = info.get("equity_usd", self.eval_env.get_attr("equity_usd")[0])
                equity_curve.append(eq)
                
                if done:
                    break
            
            # Calculate final equity
            final_equity = float(equity_curve[-1])
            
            # Log to TensorBoard
            self.logger.record("eval/final_equity", final_equity)
            self.logger.record("eval/best_equity", self.best_equity)
            
            if self.verbose >= 1:
                print(f"[Eval #{self.eval_count}] Step: {self.n_calls:,} | "
                      f"Final Equity: ${final_equity:.2f} | "
                      f"Best: ${self.best_equity:.2f}")
            
            # Check if this is the best model
            if final_equity > self.best_equity:
                self.best_equity = final_equity
                
                if self.best_model_save_path is not None:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)
                    
                    # Save the model
                    self.model.save(self.best_model_save_path)
                    
                    if self.verbose >= 1:
                        print(f"    ✅ Yeni Best Model kaydedildi: {self.best_model_save_path}")
                        print(f"    🎯 En Yüksek Equity: ${self.best_equity:.2f}")
        
        return True
