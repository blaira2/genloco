import gymnasium as gym
import numpy as np

from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback
from morphSNN import SNNFeatureExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor, VecVideoRecorder
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.logger import configure
import os


def make_pose_env_fn(base_env_fn, pose_generator, morph_vec,
                     cycle_steps=200, seed=0,
                     imitation_w = 1.0,
                     imitation_obs_indices = None):
    def _init():
        base_env = base_env_fn(seed=seed)
        base_env.reset(seed=seed)
        wrapped = MorphPhaseEnvWrapper(
            base_env=base_env,
            morph_vec=morph_vec,
            cycle_steps=cycle_steps,
            pose_generator=pose_generator,
            imitation_w=imitation_w,
            imitation_obs_indices=imitation_obs_indices,
        )
        return wrapped
    return _init


def train_ppo_with_pose_template(run_name: str,
    pose_generator,
    morph_vec,
    xml_path = "./quadruped.xml",
    timesteps = 300_000,
    parallel_envs = 4,
    initial_learning_rate = 2e-4,
    imitation_obs_indices = None,
    imitation_w = 1.0 ):

    os.makedirs("./norms", exist_ok=True)
    os.makedirs("./logs_" + run_name, exist_ok=True)
    def base_env_fn(seed=0):
        return make_quadruped_env(seed=seed, xml_path=xml_path)

    # --------- VecEnv construction ---------
    env_fns = [
        make_pose_env_fn(
            base_env_fn=base_env_fn,
            pose_generator=pose_generator,
            morph_vec=morph_vec,
            cycle_steps=200,
            seed=i,
            imitation_w=imitation_w,
            imitation_obs_indices=imitation_obs_indices,
        )
        for i in range(parallel_envs)
    ]

    if parallel_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # Optional normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    vec_env = VecMonitor(vec_env)
    vec_env.save(f"./norms/{run_name}_vecnorm.pkl")

    # --------- Logger ---------
    logger = configure(
        folder=f"./logs_{run_name}",
        format_strings=["stdout", "csv", "tensorboard"]
    )

    # --------- LR schedule ---------
    final_lr = initial_learning_rate * 0.1
    lr_schedule = LinearSchedule(
        start=initial_learning_rate,
        end=final_lr,
        end_fraction=0.9
    )

    imit_schedule = LinearSchedule(
        start=0.0,
        end=1.0,
        end_fraction=0.4
    )

    energy_schedule = LinearSchedule(
        start=0.001,
        end=0.007,
        end_fraction=0.6
    )

    morph_specs = [(xml_path, morph_vec)]
    video_cb = VideoEveryNEpisodesCallback(
        video_every=250,
        pose_generator=pose_generator,
        morph_specs=morph_specs,
        video_folder="quad_videos",
        name_prefix="quadruped_run",
        verbose=1,
        xml_path=xml_path
    )

    reward_cb = RewardDebugCallback(verbose=1)

    sched_cb = WeightScheduleCallback(
        imitation_schedule=imit_schedule,
        energy_schedule=energy_schedule,
        verbose=1
    )

    # --------- PPO model ---------
    policy_kwargs = dict(
        features_extractor_class=SNNFeatureExtractor,
        features_extractor_kwargs=dict(hidden_size=128),
        net_arch=[dict(pi=[], vf=[])]
    )

    #policy_kwargs = dict( # MLP
    #    net_arch=[128, 96, 64]  # 3 hidden layers shared by pi & vf
    #)

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=2048 // parallel_envs,
        batch_size=128,
        learning_rate=lr_schedule,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    model.set_logger(logger)

    print(f"\n🚀 Training PPO with pose templates for {run_name} ...")
    model.learn(total_timesteps=timesteps,callback=[video_cb, reward_cb, sched_cb])
    print(f"Training complete for {run_name}")

    model.save(f"{run_name}_ppo.zip")
    vec_env.close()
    return model


"""
Train a SINGLE PPO policy across multiple (xml_path, morph_vec) pairs.
        morph_specs = [
            ("./quadruped_var1.xml", var1_morph),
            ("./quadruped_var2.xml", var2_morph),
            ...
        ]
"""
def train_ppo_multi_morphologies( run_name: str,
    morph_specs,
    timesteps = 300_000,
    parallel_envs = 4,
    initial_learning_rate = 2e-4,
    cycle_steps = 200,
    pose_generator=None ):


    # Normalize inputs
    normalized_specs = []
    for xml_path, morph_vec in morph_specs:
        morph_arr = np.asarray(morph_vec, dtype=np.float32).reshape(-1)
        normalized_specs.append((xml_path, morph_arr))

    morph_specs = normalized_specs
    n_specs = len(morph_specs)

    print(f"Training PPO on {n_specs} (xml, morph_vec) configs")

    # Env factory: each worker gets one (xml, morph) in round-robin
    def make_env_fn(worker_idx: int):
        xml_path, morph_vec = morph_specs[worker_idx % n_specs]

        def _init():
            # If your make_quadruped_env returns a callable, adapt this line accordingly.
            base_env = make_quadruped_env(seed=worker_idx, xml_path=xml_path)
            env = MorphPhaseEnvWrapper(
                base_env=base_env,
                morph_vec=morph_vec,
                cycle_steps=cycle_steps,
                xml_path=xml_path,
            )
            return env

        return _init

    env_fns = [make_env_fn(i) for i in range(parallel_envs)]

    if parallel_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # VecNormalize + monitoring
    os.makedirs("./norms", exist_ok=True)
    os.makedirs(f"./logs_{run_name}", exist_ok=True)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    vec_env = VecMonitor(vec_env)
    vec_env.save(f"./norms/{run_name}_vecnorm.pkl")

    logger = configure(
        folder=f"./logs_{run_name}",
        format_strings=["stdout", "csv", "tensorboard"],
    )

    # LR schedule
    final_lr = initial_learning_rate * 0.1
    lr_schedule = LinearSchedule(
        start=initial_learning_rate,
        end=final_lr,
        end_fraction=0.9,
    )

    rep_xml, rep_morph = morph_specs[0]
    video_cb = VideoEveryNEpisodesCallback(
        video_every=250,
        pose_generator=pose_generator,
        morph_specs=morph_specs,
        video_folder="quad_videos",
        name_prefix=f"{run_name}_quad",
        verbose=1,
        xml_path=rep_xml
    )
    reward_cb = RewardDebugCallback(verbose=1)

    # SNN policy
    policy_kwargs = dict(
        features_extractor_class=SNNFeatureExtractor,
        features_extractor_kwargs=dict(hidden_size=128),
        net_arch=[dict(pi=[], vf=[])],
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=2048 // parallel_envs,
        batch_size=128,
        learning_rate=lr_schedule,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    model.set_logger(logger)

    print(f"\n🚀 Starting PPO training ({timesteps} timesteps) on {n_specs} (xml, morph) configs...")
    model.learn(total_timesteps=timesteps, callback=[video_cb, reward_cb])
    print("✅ Training complete.")

    model_path = f"{run_name}_ppo_multi.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    vec_env.close()
    return model

#helper function to merge multiple variant training runs
def merge_traj_files(paths, out_path="quadruped_morph_trajectories.npz"):
    arrays = [np.load(p) for p in paths]

    obs   = np.concatenate([a["obs"]   for a in arrays], axis=0)
    act   = np.concatenate([a["act"]   for a in arrays], axis=0)
    rew   = np.concatenate([a["rew"]   for a in arrays], axis=0)
    morph = np.concatenate([a["morph"] for a in arrays], axis=0)

    print("Merged shapes:")
    print("  obs  :", obs.shape)
    print("  act  :", act.shape)
    print("  rew  :", rew.shape)
    print("  morph:", morph.shape)

    np.savez(out_path, obs=obs, act=act, rew=rew, morph=morph)
    print(f"Saved merged dataset to {out_path}")


class MorphPhaseEnvWrapper(gym.Env):
    def __init__(self, base_env: gym.Env, morph_vec,
                 cycle_steps = 200,
                 settle_steps = 5,
                 pose_generator= None,
                 imitation_w = 1.0,
                 imitation_obs_indices=None,
                 xml_path=None):

        super().__init__()
        self.xml_path = xml_path
        self.base_env = base_env
        self.morph_vec = np.asarray(morph_vec, dtype=np.float32).reshape(-1)
        self.cycle_steps = cycle_steps
        self.settle_steps = settle_steps
        self._step_count = 0
        self._x_start = None

        # propagate render mode & metadata
        self.render_mode = getattr(base_env, "render_mode", None)
        self.metadata = getattr(base_env, "metadata", {})

        self.base_obs_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.act_dim = int(np.prod(self.base_env.action_space.shape))  # only Mujoco env actions
        base_low = self.base_env.action_space.low.astype(np.float32)
        base_high = self.base_env.action_space.high.astype(np.float32)

        a_low = np.concatenate([base_low, np.array([-1.0], dtype=np.float32)])
        a_high = np.concatenate([base_high, np.array([+1.0], dtype=np.float32)])
        self.action_space = gym.spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        self.pose_generator = pose_generator
        self.imitation_weight = imitation_w
        self.energy_weight = .007
        # which part of base_obs we compare to the template (e.g. joint angles)
        self.imitation_obs_indices = imitation_obs_indices

        #phase dimensions
        self._phase = 0.0
        self.min_phase_speed = 0.01
        self.max_phase_speed = 0.08
        morph_dim = self.morph_vec.shape[0]
        extra_dim = morph_dim + 2

        #obs
        obs_space_low = self.base_obs_space.low.astype(np.float32)
        obs_space_high = self.base_obs_space.high.astype(np.float32)

        low = np.concatenate([obs_space_low, -np.ones(extra_dim, dtype=np.float32)])
        high = np.concatenate([obs_space_high, np.ones(extra_dim, dtype=np.float32)])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        # flip detection settings
        self.flip_cos_threshold = 0.15  # < 0 means upside-down
        self._root_body_id = 1  # root body index

        # approximate sim timestep
        self.dt = getattr(self.base_env.unwrapped, "dt", 1.0 / 60.0)
        self.no_progress_time_limit = 2.5  # seconds with no progress
        self.min_progress = 0.01  # how much x must increase to count
        self._time_since_progress = 0.0
        self._last_progress_x = None


    def _compute_phase(self):
        return float(self._phase % 1.0)

    def _augment_obs(self, base_obs):
        base_obs = np.asarray(base_obs, dtype=np.float32)
        phase = self._compute_phase()
        phase_feat = np.array(
            [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)],
            dtype=np.float32
        )
        return np.concatenate([base_obs, self.morph_vec, phase_feat], axis=0)

    def set_imitation_weight(self, w: float):
        self.imitation_weight = float(w)

    def set_energy_weight(self, w: float):
        # you currently hardcode energy_w inside step(); move it to self.energy_weight first (see below)
        self.energy_weight = float(w)

    #  Get x position of the body from the underlying  env
    def _get_forward_position(self):

        try:
            data = self.base_env.unwrapped.data
            return float(data.qpos[0])  # x coordinate
        except Exception:
            return 0.0

    def _is_flipped(self):
        try:
            env = self.base_env.unwrapped
            data = env.data

            # rotation matrix for root body (id=1): 9 numbers, row-major
            # xmat[i] is a flat 3x3; columns are world-frame axes of the body
            xmat = data.xmat[self._root_body_id].reshape(3, 3)
            z_world = xmat[:, 2]  # body's local +Z in world coords

            cos_tilt = float(z_world[2])

            return cos_tilt < self.flip_cos_threshold
        except Exception:
            return False

    def reset(self, **kwargs):
        base_obs, info = self.base_env.reset(**kwargs)

        model = self.base_env.unwrapped.model
        dt = model.opt.timestep * self.base_env.unwrapped.frame_skip

        print(
            f"[Env timing] mujoco_dt={model.opt.timestep:.6f}, "
            f"frame_skip={self.base_env.unwrapped.frame_skip}, "
            f"env_dt={dt:.6f} sec"
        )

        #Physics settling
        if self.settle_steps > 0:
            zero_action = np.zeros(self.act_dim, dtype=np.float32)
            for _ in range(self.settle_steps):
                base_obs, _, terminated, truncated, _ = self.base_env.step(zero_action)
                # If env terminates during settle reset again
                if terminated or truncated:
                    base_obs, info = self.base_env.reset(**kwargs)
        if self.xml_path is not None:
            info["xml_path"] = self.xml_path
        self._step_count = 0
        self._phase = 0.0
        self._time_since_progress = 0.0
        x0 = self._get_forward_position()
        self._x_start = x0
        self._last_progress_x = x0

        return self._augment_obs(base_obs), info

    def step(self, action):
        self._step_count += 1
        #phase control
        action = np.asarray(action, dtype=np.float32)
        base_action = action[:self.act_dim]
        if action.shape[0] > self.act_dim:
            # new policy: includes phase control
            phase_ctrl = float(action[self.act_dim])
            phase_speed = self.min_phase_speed + (phase_ctrl + 1.0) * 0.5 * (
                        self.max_phase_speed - self.min_phase_speed)
        else:
            # old policy: no phase control
            phase_ctrl = 0.0
            phase_speed = 0.5 * (self.min_phase_speed + self.max_phase_speed)  # default mid-speed

        self._phase = (self._phase + phase_speed) % 1.0
        base_obs, reward, terminated, truncated, info = self.base_env.step(base_action)

        x = self._get_forward_position()
        if self._last_progress_x is None:
            self._last_progress_x = x
        if self._x_start is None:
            self._x_start = x

        delta_x = x - self._last_progress_x  # forward movement
        total_progress = x - self._x_start

        #self.energy_weight
        forward_w = 6  # applied twice: used for total_progress too
        velocity_w = 1.1
        alive_bonus = .1
        failure = 15

        action = np.asarray(action, dtype=np.float32)
        speed = delta_x / self.dt
        pos_speed = max(speed, 0)
        energy_penalty = self.energy_weight * float(np.sum(base_action ** 2))
        forward_reward = forward_w * delta_x
        log_vel_reward = velocity_w * np.log(1.0 + pos_speed)
        if speed < 0:
            log_vel_reward += speed

        # reward without pose
        reward = log_vel_reward - energy_penalty

        # Imitation reward
        if self.pose_generator is not None and self.imitation_weight != 0.0:
            # phase in [0,1)
            phase = self._compute_phase()  # uses step_count % cycle_steps

            # pose_generator should return a *base_env-style pose* for this morph & phase
            target_pose = self.pose_generator(self.morph_vec, phase)
            target_pose = np.asarray(target_pose, dtype=np.float32)
            current_pose = np.asarray(base_obs, dtype=np.float32)

            # joint angles only
            if self.imitation_obs_indices is not None:
                current_pose = current_pose[self.imitation_obs_indices]
                if target_pose.shape[0] == len(self.imitation_obs_indices):
                    pass
                else:
                    target_pose = target_pose[self.imitation_obs_indices]

            # make sure same length (in case of slight mismatch)
            L = min(len(current_pose), len(target_pose))
            if L > 0:
                diff = current_pose[:L] - target_pose[:L]
                pose_mse = float(np.mean(diff ** 2))

                # Imitation reward: higher when MSE is small.
                imitation_reward = - self.imitation_weight * pose_mse
                reward += imitation_reward
                info["imitation_reward"] = imitation_reward
                info["pose_mse"] = pose_mse

        # no-forward-progress termination
        info["fail_penalty"] = 0
        info["xml_path"] = self.xml_path

        if delta_x > self.min_progress:
            self._last_progress_x = x
            self._time_since_progress = 0.0
            reward += alive_bonus
        else:
            self._time_since_progress += self.dt
            if self._time_since_progress >= self.no_progress_time_limit:
                terminated = True
                reward -= failure  # extra penalty
                info = dict(info)
                info["fail_penalty"] += failure
        # flipped / fallen termination
        if self._is_flipped():
            terminated = True
            reward -= failure
            info = dict(info)
            info["fail_penalty"] += failure
        if terminated or truncated:
            forward_reward += forward_w * total_progress
        reward += forward_reward

        info["forward_reward"] = forward_reward
        info["energy_penalty"] = energy_penalty
        info["alive_reward"] = alive_bonus
        info["velocity_reward"] = log_vel_reward
        info["phase_speed"] = phase_speed

        info["delta_x"] = delta_x

        return self._augment_obs(base_obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()



class VideoEveryNEpisodesCallback(BaseCallback):
    def __init__( self,
        video_every: int,
        pose_generator,
        morph_specs,
        video_folder = "videos",
        name_prefix = "run",
        verbose: int = 0,
        xml_path = "./quadruped.xml"):

        super().__init__(verbose)
        self.video_every = video_every
        self.env_fn = lambda: make_quadruped_env_video(xml_path=xml_path)
        self.video_folder = video_folder
        self.name_prefix = name_prefix

        self.pose_generator = pose_generator
        self.morph_specs = [
            (xml_path, np.asarray(morph_vec, dtype=np.float32).reshape(-1))
            for (xml_path, morph_vec) in morph_specs
        ]
        self.spec_idx = 0
        self.episode_count = 0

        os.makedirs(video_folder, exist_ok=True)


    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode"):
                self.episode_count += 1
                if self.episode_count % self.video_every == 0:
                    self.record_video()
        return True

    def record_video(self):
        if self.verbose:
            print(f"\n🎥 Recording episode {self.episode_count}...")

        # Pick current (xml, morph) and advance index
        xml_path, morph_vec = self.morph_specs[self.spec_idx]
        self.spec_idx = (self.spec_idx + 1) % len(self.morph_specs)

        #fresh base env with rgb_array for  xml
        base_env = make_quadruped_env_video(xml_path=xml_path)

        # wrap it like in training
        wrapped_env = MorphPhaseEnvWrapper(
            base_env=base_env,
            morph_vec=morph_vec
        )

        env = RecordVideo(
            wrapped_env,
            video_folder=self.video_folder,
            name_prefix=f"{self.name_prefix}_ep{self.episode_count}",
            episode_trigger=lambda ep: True,
        )

        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)

        env.close()
        if self.verbose:
            print("Video recording complete.")


def make_quadruped_env(seed=0, xml_path = "./quadruped.xml"):

    env = gym.make("Ant-v5", xml_file=xml_path)
    env.reset(seed=seed)
    return env


def make_quadruped_env_video(xml_path = "./quadruped.xml"):

    env = gym.make("Ant-v5", xml_file=xml_path, render_mode="rgb_array")
    return env

#run a trained policy to collect trajectories
def collect_quadruped_morph_trajectories(ppo_path: str,
    morph_specs,
    episodes_per_morph: int = 20,
    max_steps_per_ep: int = 1000,
    out_path: str = "quadruped_morph_trajectories.npz",
    deterministic: bool = False ):

    print(f"Loading PPO model from {ppo_path} ...")
    model = PPO.load(ppo_path)

    all_obs   = []
    all_act   = []
    all_rew   = []
    all_morph = []
    all_ep_id = []  # global episode id per step
    all_t_in_ep = []  # timestep within that episode per step
    ep_global = 0

    for m_idx, (xml_path, morph_vec) in enumerate(morph_specs):
        print(f"\n=== Collecting for morph {m_idx}: {morph_vec} ===")

        for ep in range(episodes_per_morph):
            # base Ant env
            base_env = make_quadruped_env(seed=ep, xml_path=xml_path)
            # wrap with phase + morph features
            env = MorphPhaseEnvWrapper(
                base_env=base_env,
                morph_vec=morph_vec,
                cycle_steps=200,
            )

            obs, info = env.reset()
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < max_steps_per_ep:
                # log obs & morph at each step
                all_obs.append(obs.astype(np.float32))
                all_morph.append(np.asarray(morph_vec, dtype=np.float32))

                all_ep_id.append(ep_global)
                all_t_in_ep.append(step)

                # policy action
                action, _ = model.predict(obs, deterministic=deterministic)
                next_obs, reward, done, truncated, info = env.step(action)

                all_act.append(action.astype(np.float32))
                all_rew.append(float(reward))

                obs = next_obs
                step += 1

            env.close()
            print(f"  Morph {m_idx}, episode {ep} finished after {step} steps.")

    obs   = np.vstack(all_obs)        # (T, obs_dim)
    act   = np.vstack(all_act)        # (T, act_dim)
    rew   = np.asarray(all_rew)
    morph = np.vstack(all_morph)      # (T, morph_dim)
    ep_id = np.asarray(all_ep_id, dtype=np.int32)
    t_in_ep = np.asarray(all_t_in_ep, dtype=np.int32)

    print("\nFinal shapes:")
    print("  obs  :", obs.shape)
    print("  act  :", act.shape)
    print("  rew  :", rew.shape)
    print("  morph:", morph.shape)
    print("  ep_id :", ep_id.shape)
    print("  t_in_ep:", t_in_ep.shape)

    np.savez(out_path, obs=obs, act=act, rew=rew, morph=morph, ep_id=ep_id, t_in_ep=t_in_ep)
    print(f"Saved trajectories to {out_path}")

class WeightScheduleCallback(BaseCallback):
    def __init__(self, imitation_schedule, energy_schedule, verbose=0):
        super().__init__(verbose)
        self.imitation_schedule = imitation_schedule
        self.energy_schedule = energy_schedule

    def _on_step(self) -> bool:
        # progress_remaining ∈ [1, 0]
        progress = self.model._current_progress_remaining

        imit_w = float(self.imitation_schedule(progress))
        energy_w = float(self.energy_schedule(progress))

        # unwrap VecEnv → Dummy/Subproc → gym env
        vec = self.training_env
        while hasattr(vec, "venv"):
            vec = vec.venv

        if hasattr(vec, "envs"):  # DummyVecEnv
            for e in vec.envs:
                env = e
                while hasattr(env, "env"):
                    env = env.env
                env.imitation_weight = imit_w
                env.energy_weight = energy_w
        else:
            vec.env_method("set_imitation_weight", imit_w)
            vec.env_method("set_energy_weight", energy_w)

        if self.verbose and self.n_calls % 2048 == 0:
            print(f"[sched] imitation_w={imit_w:.3f}, energy_w={energy_w:.5f}")

        return True

class RewardDebugCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_sums = None
        self.ep_len = None
        self.ep_xml = None  # track xml path per env

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        # numeric episode sums
        self.ep_sums = []
        self.ep_len = [0] * n_envs
        # xml path per env (string)
        self.ep_xml = ["" for _ in range(n_envs)]

        for _ in range(n_envs):
            self.ep_sums.append({
                "total": 0.0,
                "forward": 0.0,
                "alive_reward": 0.0,
                "energy_penalty": 0.0,
                "velocity_reward": 0.0,
                "imitation_reward": 0.0,
                "fail_penalty": 0.0,
                "phase_speed": 0.0,
            })

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape (n_envs,)
        infos = self.locals["infos"]      # list of dicts
        dones = self.locals["dones"]      # shape (n_envs,)

        n_envs = len(dones)
        for i in range(n_envs):
            info = infos[i]
            s = self.ep_sums[i]

            # accumulate numeric stats
            s["total"] += float(rewards[i])
            s["forward"] += info.get("forward_reward", 0.0)
            s["alive_reward"] += info.get("alive_reward", 0.0)
            s["fail_penalty"] += info.get("fail_penalty", 0.0)
            s["velocity_reward"] += info.get("velocity_reward", 0.0)
            s["imitation_reward"] += info.get("imitation_reward", 0.0)
            s["energy_penalty"] += info.get("energy_penalty", 0.0)
            s["phase_speed"] += info.get("phase_speed", 0.0)


            # update xml path if present
            if "xml_path" in info:
                self.ep_xml[i] = info["xml_path"]

            self.ep_len[i] += 1

            if dones[i]:
                L = self.ep_len[i] or 1
                xml = self.ep_xml[i] or "unknown"

                print(
                    f"[env {i}] ep_len={L:4d} | "
                    f"R_mean={s['total'] / L: .3f} | "
                    f"fwd={s['forward'] / L: .3f} | "
                    f"vel={s['velocity_reward'] / L: .3f} | "
                    f"imitat={s['imitation_reward'] / L: .3f} | "
                    f"alive={s['alive_reward'] / L: .3f} | "
                    f"energy_p={s['energy_penalty'] / L: .3f} | "
                    f"fail={s['fail_penalty'] / L: .3f} | "
                    f"phase_spd={s['phase_speed'] / L: .3f} | "
                    f"var={xml}"
                )

                # reset for next episode
                self.ep_sums[i] = {k: 0.0 for k in s}
                self.ep_len[i] = 0
                self.ep_xml[i] = ""

        return True