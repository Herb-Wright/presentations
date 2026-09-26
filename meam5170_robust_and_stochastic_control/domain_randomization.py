#!/usr/bin/env python3
"""PPO domain-randomization example for the MEAM 5170 presentation.

Two HalfCheetah-v5 policies are trained with identical PPO settings across 32
independent simulators. The nominal policy sees the default MuJoCo model,
while the final DR policy randomizes actuator gear and absolute floor/foot
sliding friction at every episode reset; mass, damping, and gravity remain
nominal. The actor receives
only the original observation; a privileged critic additionally receives the
24-vector [8 body-mass scales, 3 friction scales, 6 gear scales, 6 damping
scales, gravity scale]. Both policies are
evaluated on both objectives and the results are rendered as a compact,
slide-friendly figure.

The default run is suitable for a presentation (2,097,152 steps per policy).
``--fast`` is a quick smoke-test/presentation draft (8k steps per policy).
Requires ``stable-baselines3``, ``gymnasium[mujoco]``, ``mujoco``, seaborn,
and matplotlib.  Install in the project venv with::

    uv pip install --python .venv/bin/python 'stable-baselines3>=2.4' \\
        'gymnasium[mujoco]>=1.0' 'mujoco>=3.1'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Keep Matplotlib's font/cache files in a writable scratch location on the
# managed presentation workspace.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/meam5170-matplotlib")
# MuJoCo's RGB panel should work on headless CI/containers as well as desktop
# machines. Users with a display can override this before launching.
os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import gymnasium as gym
    from gymnasium import Wrapper
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    from stable_baselines3.common.torch_layers import MlpExtractor
    import torch as th
    import torch.nn as nn
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - actionable CLI error
    raise SystemExit(
        "Missing RL dependencies. Install with:\n"
        "  uv pip install --python .venv/bin/python "
        "'stable-baselines3>=2.4' 'gymnasium[mujoco]>=1.0' 'mujoco>=3.1'"
    ) from exc

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import seaborn as sns


# Final candidate A: actuator gear and absolute floor/foot sliding friction
# are randomized; body mass, joint damping, and gravity remain nominal.
BODY_MASS_RANGE = (1.00, 1.00)
BODY_MASS_RESIDUAL_RANGE = (1.00, 1.00)
GEAR_RANGE = (0.50, 1.50)
GEAR_RESIDUAL_RANGE = (0.90, 1.10)
DAMPING_RANGE = (1.00, 1.00)
FRICTION_RANGE = (0.20, 1.00)
GRAVITY_RANGE = (1.00, 1.00)
FRICTION_IS_ABSOLUTE = True
ENV_ID = "HalfCheetah-v5"
DEFAULT_STEPS = 2_097_152
DEFAULT_EVAL_FREQ = 262_144
DEFAULT_EPISODES = 64
CACHE_VERSION = 7
N_TRAIN_ENVS = 32
PPO_ROLLOUT_STEPS = 128
TRAIN_ROLLOUT_SAMPLES = N_TRAIN_ENVS * PPO_ROLLOUT_STEPS

# Named presets make the empirical search reproducible. The default is the
# adopted final distribution; future candidates can be added without
# changing wrapper logic or evaluation bookkeeping.
DISTRIBUTION_PRESETS = {
    "broad_previous": {
        "body_mass": (0.90, 1.10), "body_mass_residual": (0.95, 1.05),
        "gear": (0.75, 1.25), "gear_residual": (0.85, 1.15),
        "damping": (0.50, 2.00), "friction": (0.15, 0.80), "gravity": (0.95, 1.05),
    },
    "candidate_a": {
        "body_mass": (1.00, 1.00), "body_mass_residual": (1.00, 1.00),
        "gear": (0.50, 1.50), "gear_residual": (0.90, 1.10),
        "damping": (1.00, 1.00), "friction": (0.20, 1.00),
        "gravity": (1.00, 1.00), "friction_absolute": True,
    },
    "candidate_b": {
        "body_mass": (1.00, 1.00), "body_mass_residual": (1.00, 1.00),
        "gear": (0.30, 1.50), "gear_residual": (1.00, 1.00),
        "damping": (1.00, 1.00), "friction": (0.20, 1.00),
        "gravity": (1.00, 1.00), "friction_absolute": True,
    },
    "candidate_c": {
        "body_mass": (1.00, 1.00), "body_mass_residual": (1.00, 1.00),
        "gear": (0.50, 1.50), "gear_residual": (0.90, 1.10),
        "damping": (0.50, 2.00), "friction": (0.20, 1.00),
        "gravity": (1.00, 1.00), "friction_absolute": True,
    },
}


def apply_distribution(name: str) -> None:
    """Set wrapper globals from a named, metadata-serializable preset."""
    if name not in DISTRIBUTION_PRESETS:
        raise ValueError(f"Unknown distribution {name!r}; choose from {sorted(DISTRIBUTION_PRESETS)}")
    config = DISTRIBUTION_PRESETS[name]
    global BODY_MASS_RANGE, BODY_MASS_RESIDUAL_RANGE, GEAR_RANGE, GEAR_RESIDUAL_RANGE
    global DAMPING_RANGE, FRICTION_RANGE, GRAVITY_RANGE, FRICTION_IS_ABSOLUTE
    BODY_MASS_RANGE = tuple(config["body_mass"])
    BODY_MASS_RESIDUAL_RANGE = tuple(config["body_mass_residual"])
    GEAR_RANGE = tuple(config["gear"])
    GEAR_RESIDUAL_RANGE = tuple(config["gear_residual"])
    DAMPING_RANGE = tuple(config["damping"])
    FRICTION_RANGE = tuple(config["friction"])
    GRAVITY_RANGE = tuple(config["gravity"])
    FRICTION_IS_ABSOLUTE = bool(config.get("friction_absolute", False))


class HalfCheetahDomainRandomization(Wrapper):
    """Episode-wise MuJoCo domain randomization with safe restoration.

    Candidate A draws nominal mass, damping, and gravity; global actuator gear
    U[0.5, 1.5] times per-actuator residuals U[0.9, 1.1]; and absolute
    floor/foot sliding friction U[0.2, 1.0]. Copies of nominal arrays are restored before
    each draw; no topology changes are made.
    """

    def __init__(self, env: gym.Env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        model = self.unwrapped.model
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_geom_friction = model.geom_friction.copy()
        self._nominal_actuator_gear = model.actuator_gear.copy()
        self._nominal_dof_damping = model.dof_damping.copy()
        self._nominal_gravity = model.opt.gravity.copy()
        self._actuated_dof_indices = np.concatenate(
            [np.arange(model.jnt_dofadr[j], model.jnt_dofadr[j] +
                       (3 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1))
             for j in range(3, model.njnt)]
        )
        self._friction_geom_indices = [i for i in range(model.ngeom)
                                       if model.geom(i).name in {"floor", "bfoot", "ffoot"}]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        model = self.unwrapped.model
        model.body_mass[:] = self._nominal_body_mass
        model.geom_friction[:] = self._nominal_geom_friction
        model.actuator_gear[:] = self._nominal_actuator_gear
        model.dof_damping[:] = self._nominal_dof_damping
        model.opt.gravity[:] = self._nominal_gravity
        mujoco.mj_setConst(model, self.unwrapped.data)
        params = {"body_mass": 1.0, "geom_friction": 1.0, "gear": 1.0, "damping": 1.0, "gravity": 1.0}
        if self.enabled:
            # gymnasium's reset(seed=...) initializes self.np_random before
            # this call, making evaluation episodes reproducible.
            if seed is not None:
                super().reset(seed=seed)
            rng = self.np_random
            params = {
                "body_mass": float(rng.uniform(*BODY_MASS_RANGE)) * rng.uniform(*BODY_MASS_RESIDUAL_RANGE, size=self._nominal_body_mass.shape),
                "geom_friction": rng.uniform(*FRICTION_RANGE, size=len(self._friction_geom_indices)),
                "gear": float(rng.uniform(*GEAR_RANGE)) * rng.uniform(*GEAR_RESIDUAL_RANGE, size=self._nominal_actuator_gear.shape[0]),
                "damping": np.exp(rng.uniform(np.log(DAMPING_RANGE[0]), np.log(DAMPING_RANGE[1]), size=len(self._actuated_dof_indices))),
                "gravity": float(rng.uniform(*GRAVITY_RANGE)),
            }
            model.body_mass[:] = self._nominal_body_mass * params["body_mass"]
            for index, scale in zip(self._friction_geom_indices, params["geom_friction"]):
                model.geom_friction[index, 0] = scale if FRICTION_IS_ABSOLUTE else self._nominal_geom_friction[index, 0] * scale
            model.actuator_gear[:] = self._nominal_actuator_gear * params["gear"][:, None]
            model.dof_damping[self._actuated_dof_indices] = (
                self._nominal_dof_damping[self._actuated_dof_indices] * params["damping"]
            )
            # Gravity is a signed vector; scale its magnitude without changing
            # its direction (MuJoCo's default is [0, 0, -9.81]).
            model.opt.gravity[:] = self._nominal_gravity * params["gravity"]
            # Body mass affects derived inertial constants. Recompute them
            # before Gymnasium's reset/forward pass uses the modified model.
            mujoco.mj_setConst(model, self.unwrapped.data)
        obs, info = self.env.reset(seed=None if (self.enabled and seed is not None) else seed, options=options)
        info = dict(info)
        body_scale = np.divide(model.body_mass, self._nominal_body_mass,
                               out=np.ones_like(model.body_mass), where=self._nominal_body_mass != 0)
        friction_scale = np.divide(
            model.geom_friction[self._friction_geom_indices, 0],
            self._nominal_geom_friction[self._friction_geom_indices, 0],
            out=np.ones(len(self._friction_geom_indices)),
            where=self._nominal_geom_friction[self._friction_geom_indices, 0] != 0,
        )
        gear_scale = np.divide(model.actuator_gear[:, 0], self._nominal_actuator_gear[:, 0],
                               out=np.ones(model.nu), where=self._nominal_actuator_gear[:, 0] != 0)
        damping_scale = np.divide(model.dof_damping[self._actuated_dof_indices],
                                  self._nominal_dof_damping[self._actuated_dof_indices],
                                  out=np.ones(len(self._actuated_dof_indices)),
                                  where=self._nominal_dof_damping[self._actuated_dof_indices] != 0)
        gravity_scale = float(model.opt.gravity[2] / self._nominal_gravity[2])
        info["domain_vector"] = np.concatenate((body_scale, friction_scale, gear_scale,
                                                   damping_scale, [gravity_scale])).astype(np.float32)
        info["domain_parameters"] = {"body_mass_scale_mean": float(np.mean(body_scale)) if self.enabled else 1.0,
                                      "geom_friction_scale_mean": float(np.mean(params["geom_friction"])) if self.enabled else 1.0,
                                      "gravity_scale": params["gravity"]}
        return obs, info


class PrivilegedDomainObservation(gym.ObservationWrapper):
    """Dict observation: actor gets ``observation``; critic also gets ``domain``."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "domain": spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32),
        })
        self._domain = np.ones(24, dtype=np.float32)

    def observation(self, observation):
        return {"observation": np.asarray(observation, dtype=np.float32), "domain": self._domain.copy()}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._domain = np.asarray(info.get("domain_vector", np.ones(24)), dtype=np.float32)
        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "domain_vector" in info:
            self._domain = np.asarray(info["domain_vector"], dtype=np.float32)
        return self.observation(observation), reward, terminated, truncated, info


class PrivilegedMlpExtractor(nn.Module):
    """Separate actor/value MLPs; only the value MLP receives domain features."""

    def __init__(self, total_dim: int, actor_dim: int, activation_fn: type[nn.Module]):
        super().__init__()
        self.policy_net = nn.Sequential(nn.Linear(actor_dim, 64), activation_fn(), nn.Linear(64, 64), activation_fn())
        self.value_net = nn.Sequential(nn.Linear(total_dim, 64), activation_fn(), nn.Linear(64, 64), activation_fn())
        self.latent_dim_pi = self.latent_dim_vf = 64

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        # Gymnasium sorts Dict keys alphabetically, so CombinedExtractor emits
        # domain (24) before observation (17); take the physical suffix.
        return self.policy_net(features[..., -self.policy_net[0].in_features :])

    def forward_critic(self, features):
        return self.value_net(features)


class PrivilegedCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 policy with actor invariant to the privileged domain dictionary key."""

    def _build_mlp_extractor(self):
        actor_dim = int(np.prod(self.observation_space["observation"].shape))
        self.mlp_extractor = PrivilegedMlpExtractor(self.features_dim, actor_dim, self.activation_fn)


def make_env(*, randomized: bool, render_mode: str | None = None, seed: int | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = HalfCheetahDomainRandomization(env, enabled=randomized)
    env = PrivilegedDomainObservation(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_training_vec_env(*, randomized: bool, seed: int) -> DummyVecEnv:
    """Build 32 independent simulators; each reset draws its own domain."""
    env_fns = [lambda i=i: make_env(randomized=randomized, seed=seed + i) for i in range(N_TRAIN_ENVS)]
    return DummyVecEnv(env_fns)


def evaluate(model: PPO, *, randomized: bool, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return returns, lengths, and sampled domain summaries on paired seeds."""
    env = make_env(randomized=randomized)
    returns, lengths, parameters = [], [], []
    try:
        for episode_seed in seeds:
            obs, info = env.reset(seed=episode_seed)
            params = info.get("domain_parameters", {})
            parameters.append([float(params.get("body_mass_scale_mean", 1.0)),
                               float(params.get("geom_friction_scale_mean", 1.0)),
                               float(params.get("gravity_scale", 1.0))])
            done = False
            total, length = 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                done = bool(terminated or truncated)
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    return np.asarray(returns), np.asarray(lengths), np.asarray(parameters)


def run_privileged_self_test(seed: int = 5170) -> None:
    """Verify actor invariance and critic sensitivity to the domain key."""
    env = make_env(randomized=True, seed=seed)
    model = PPO(PrivilegedCriticPolicy, env, seed=seed, n_steps=128, batch_size=128, device="cpu", verbose=0)
    obs, _ = env.reset(seed=seed)
    altered = {key: value.copy() for key, value in obs.items()}
    altered["domain"] = altered["domain"] + 0.37
    obs_a, _ = model.policy.obs_to_tensor(obs)
    obs_b, _ = model.policy.obs_to_tensor(altered)
    with th.no_grad():
        feat_a = model.policy.extract_features(obs_a)
        feat_b = model.policy.extract_features(obs_b)
        actor_a = model.policy.mlp_extractor.forward_actor(feat_a)
        actor_b = model.policy.mlp_extractor.forward_actor(feat_b)
        critic_a = model.policy.mlp_extractor.forward_critic(feat_a)
        critic_b = model.policy.mlp_extractor.forward_critic(feat_b)
    action_a, _ = model.predict(obs, deterministic=True)
    action_b, _ = model.predict(altered, deterministic=True)
    env.close()
    actor_delta = float(th.max(th.abs(actor_a - actor_b)))
    critic_delta = float(th.max(th.abs(critic_a - critic_b)))
    action_delta = float(np.max(np.abs(action_a - action_b)))
    if actor_delta > 1e-7 or action_delta > 1e-7 or critic_delta <= 1e-7:
        raise AssertionError(f"privileged critic self-test failed: actor Δ={actor_delta}, action Δ={action_delta}, critic Δ={critic_delta}")
    print(f"Privileged critic self-test passed: actor Δ={actor_delta:.2e}, action Δ={action_delta:.2e}, critic Δ={critic_delta:.2e}")


def train_and_measure(args: argparse.Namespace) -> dict[str, Any]:
    # PPO collects complete rollouts. Round requested checkpoints up to the
    # next rollout boundary so labels are actual SB3 environment steps.
    requested = np.unique(np.r_[np.arange(args.eval_freq, args.timesteps + 1, args.eval_freq), args.timesteps])
    checkpoints = np.unique(np.ceil(requested / TRAIN_ROLLOUT_SAMPLES).astype(int) * TRAIN_ROLLOUT_SAMPLES)
    print(f"Requested {args.timesteps:,} steps; PPO rollout-aligned schedule ends at {checkpoints[-1]:,} actual steps")
    eval_seeds = [args.seed + 10_000 + i for i in range(args.eval_episodes)]
    cache_path = args.output.with_suffix(".npz")
    meta = {"cache_version": CACHE_VERSION, "distribution": args.distribution,
            "timesteps": args.timesteps, "actual_timesteps": int(checkpoints[-1]),
            "eval_freq": args.eval_freq, "eval_episodes": args.eval_episodes, "seed": args.seed,
            "ppo_rollout_steps": PPO_ROLLOUT_STEPS, "n_train_envs": N_TRAIN_ENVS,
            "train_rollout_samples": TRAIN_ROLLOUT_SAMPLES,
            "body_mass_range": BODY_MASS_RANGE, "body_mass_residual_range": BODY_MASS_RESIDUAL_RANGE,
            "gear_range": GEAR_RANGE, "gear_residual_range": GEAR_RESIDUAL_RANGE,
            "damping_range": DAMPING_RANGE, "friction_is_absolute": FRICTION_IS_ABSOLUTE,
            "friction_range": FRICTION_RANGE,
            "gravity_range": GRAVITY_RANGE}
    if not args.no_cache and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if json.loads(str(cached["meta"])) == meta:
                print(f"Using cached training/evaluation data: {cache_path}")
                return {key: cached[key] for key in cached.files if key != "meta"}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    results: dict[str, Any] = {"steps": checkpoints, "meta": meta}
    # Train both variants with exactly the same PPO hyperparameters and seeds.
    for policy_name, randomized in (("nominal", False), ("domain_randomized", True)):
        train_env = make_training_vec_env(randomized=randomized, seed=args.seed)
        model = PPO(
            PrivilegedCriticPolicy, train_env, seed=args.seed, verbose=0, n_steps=PPO_ROLLOUT_STEPS,
            batch_size=256, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.0, clip_range=0.2, device="cpu",
        )
        dr_mean, dr_std, nom_mean, nom_std = [], [], [], []
        dr_returns, nom_returns, dr_parameters, nom_parameters = [], [], [], []
        completed = 0
        for checkpoint in checkpoints:
            model.learn(total_timesteps=int(checkpoint - completed), reset_num_timesteps=False, progress_bar=False)
            completed = int(checkpoint)
            dr, _, dr_params = evaluate(model, randomized=True, seeds=eval_seeds)
            nom, _, nom_params = evaluate(model, randomized=False, seeds=eval_seeds)
            dr_returns.append(dr); nom_returns.append(nom)
            dr_parameters.append(dr_params); nom_parameters.append(nom_params)
            dr_mean.append(dr.mean()); dr_std.append(dr.std(ddof=1) if len(dr) > 1 else 0.0)
            nom_mean.append(nom.mean()); nom_std.append(nom.std(ddof=1) if len(nom) > 1 else 0.0)
            print(f"{policy_name:17s} {checkpoint:>8,d} steps | DR {dr.mean():8.1f} | nominal {nom.mean():8.1f}")
        train_env.close()
        prefix = "dr_policy" if randomized else "nom_policy"
        results[f"{prefix}_dr_mean"] = np.asarray(dr_mean)
        results[f"{prefix}_dr_std"] = np.asarray(dr_std)
        results[f"{prefix}_nom_mean"] = np.asarray(nom_mean)
        results[f"{prefix}_nom_std"] = np.asarray(nom_std)
        results[f"{prefix}_dr_returns"] = np.asarray(dr_returns)
        results[f"{prefix}_nom_returns"] = np.asarray(nom_returns)
        results[f"{prefix}_dr_parameters"] = np.asarray(dr_parameters)
        results[f"{prefix}_nom_parameters"] = np.asarray(nom_parameters)
    if not args.no_cache:
        np.savez_compressed(cache_path, meta=json.dumps(meta), **{k: v for k, v in results.items() if k != "meta"})
    return results


def make_figure(data: dict[str, Any], *, output: Path, seed: int) -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="deep", rc={"font.family": "Roboto"})
    colors = {"Nominal PPO": sns.color_palette("deep")[0], "DR PPO": sns.color_palette("deep")[2]}
    # A single row is deliberately close to the 3:1 aspect ratio of a slide
    # banner, with the rendered environment occupying the first panel.
    fig = plt.figure(figsize=(17.2, 5.8), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, left=0.025, right=0.99, bottom=0.16, top=0.82,
                          width_ratios=[1.0, 1.25, 1.25], wspace=0.20)
    ax_img = fig.add_subplot(gs[0, 0])
    render_env = make_env(randomized=False, render_mode="rgb_array", seed=seed + 99)
    try:
        image: AxesImage = ax_img.imshow(render_env.render())
    finally:
        render_env.close()
    ax_img.set_title("HalfCheetah-v5", pad=10, fontweight="normal", loc="center")
    ax_img.text(0.03, 0.04, "MuJoCo benchmark", transform=ax_img.transAxes, color="white", fontsize=13,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 4})
    ax_img.axis("off")

    x = data["steps"] / 1000.0
    panels = [(fig.add_subplot(gs[0, 1]), "Domain-randomized objective", "_dr"),
              (fig.add_subplot(gs[0, 2]), "Nominal objective", "_nom")]
    for ax, title, suffix in panels:
        for label, prefix in (("Nominal PPO", "nom_policy"), ("DR PPO", "dr_policy")):
            mean = data[f"{prefix}{suffix}_mean"]
            std = data[f"{prefix}{suffix}_std"]
            ax.plot(x, mean, label=label, color=colors[label], linewidth=2.5, marker="o", markersize=4)
            ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.12, linewidth=0)
        ax.set_title(title, loc="center", fontweight="normal", pad=8)
        ax.set_xlabel("Environment steps (×10³)")
        ax.set_ylabel("Episode return")
        ax.axhline(0, color="0.35", linewidth=0.8, zorder=0)
        ax.set_xlim(left=0)
        ax.legend(frameon=True, fontsize=12, loc="best")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_STEPS, help="PPO steps per policy")
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ, help="Evaluate every N training steps")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per objective/checkpoint")
    parser.add_argument("--seed", type=int, default=5170, help="Shared training seed and base eval seed")
    parser.add_argument("--distribution", choices=sorted(DISTRIBUTION_PRESETS), default="candidate_a",
                        help="Named domain-randomization preset")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("domain_randomization.png"), help="PNG output path")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write the adjacent .npz result cache")
    parser.add_argument("--self-test-privileged", action="store_true",
                        help="Verify actor invariance and critic sensitivity, then exit")
    parser.add_argument("--fast", action="store_true", help="Quick draft: 8k steps, 2 evaluations per checkpoint")
    args = parser.parse_args()
    if args.fast:
        args.timesteps, args.eval_freq, args.eval_episodes = 8_192, 4_096, 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.timesteps < 1024 or args.eval_freq < 1024:
        parser.error(f"--timesteps and --eval-freq must each be at least one vector rollout={TRAIN_ROLLOUT_SAMPLES}")
    apply_distribution(args.distribution)
    if args.self_test_privileged:
        run_privileged_self_test(args.seed)
        return
    data = train_and_measure(args)
    make_figure(data, output=args.output, seed=args.seed)
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
