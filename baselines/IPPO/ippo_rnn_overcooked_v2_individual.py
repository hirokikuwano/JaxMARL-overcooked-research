import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = (
            config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        print("Update steps: ", update_steps)
        print("Warmup epochs: ", warmup_steps)
        print("Cosine epochs: ", cosine_epochs)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng):

        # INIT NETWORK — Individual Weight: 2つのネットワークを別々の乱数で初期化
        network_0 = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        network_1 = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        rng, _rng_0, _rng_1 = jax.random.split(rng, 3)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        network_params_0 = network_0.init(_rng_0, init_hstate, init_x)
        network_params_1 = network_1.init(_rng_1, init_hstate, init_x)

        # ===== 検証1: ネットワーク初期化 =====
        print("=" * 60)
        print("[検証1] ネットワーク初期化")
        print(f"  init_x obs shape: {init_x[0].shape}")  # (1, 256, 5, 5, 26) を期待
        print(f"  init_x done shape: {init_x[1].shape}")  # (1, 256) を期待
        print(f"  init_hstate shape: {init_hstate.shape}")  # (256, 128) を期待
        # パラメータが別々の乱数で初期化されているか確認
        flat_params_0 = jax.tree_util.tree_leaves(network_params_0)
        flat_params_1 = jax.tree_util.tree_leaves(network_params_1)
        params_same = jnp.all(jnp.array([jnp.all(a == b) for a, b in zip(flat_params_0, flat_params_1)]))
        jax.debug.print("  network_0 と network_1 のパラメータが同一: {s}", s=params_same)  # False を期待
        print(f"  パラメータ数: {sum(p.size for p in flat_params_0)}")
        print("=" * 60)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state_0 = TrainState.create(
            apply_fn=network_0.apply,
            params=network_params_0,
            tx=tx,
        )
        train_state_1 = TrainState.create(
            apply_fn=network_1.apply,
            params=network_params_1,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate_0 = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        init_hstate_1 = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        # ===== 検証2: hstate・done 初期化 =====
        print("\n" + "=" * 60)
        print("[検証2] hstate・done 初期化")
        print(f"  init_hstate_0 shape: {init_hstate_0.shape}")  # (256, 128) を期待
        print(f"  init_hstate_1 shape: {init_hstate_1.shape}")  # (256, 128) を期待
        print(f"  初期 done shape: ({config['NUM_ENVS']},)")  # (256,) を期待
        print("=" * 60)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state_0, train_state_1,
                    env_state, last_obs, last_done, update_step,
                    hstate_0, hstate_1, rng,
                ) = runner_state

                # SELECT ACTION — agent ごとに独立して処理
                rng, _rng_0, _rng_1 = jax.random.split(rng, 3)

                obs_0 = last_obs["agent_0"]  # (256, 5, 5, 26)
                obs_1 = last_obs["agent_1"]  # (256, 5, 5, 26)

                ac_in_0 = (obs_0[np.newaxis, :], last_done[np.newaxis, :])
                ac_in_1 = (obs_1[np.newaxis, :], last_done[np.newaxis, :])

                hstate_0, pi_0, value_0 = network_0.apply(train_state_0.params, hstate_0, ac_in_0)
                hstate_1, pi_1, value_1 = network_1.apply(train_state_1.params, hstate_1, ac_in_1)

                action_0 = pi_0.sample(seed=_rng_0)
                action_1 = pi_1.sample(seed=_rng_1)
                log_prob_0 = pi_0.log_prob(action_0)
                log_prob_1 = pi_1.log_prob(action_1)

                # ===== 検証3: _env_step 内の shape 確認（最初の1回だけ） =====
                def _debug_print_env_step():
                    jax.debug.print("=" * 60)
                    jax.debug.print("[検証3] _env_step 内 shape 確認")
                    jax.debug.print("  obs_0 shape: {s}", s=obs_0.shape)
                    jax.debug.print("  obs_1 shape: {s}", s=obs_1.shape)
                    jax.debug.print("  last_done shape: {s}", s=last_done.shape)
                    jax.debug.print("  hstate_0 shape: {s}", s=hstate_0.shape)
                    jax.debug.print("  value_0 shape: {s}", s=value_0.shape)
                    jax.debug.print("  action_0 shape: {s}", s=action_0.shape)
                    jax.debug.print("  log_prob_0 shape: {s}", s=log_prob_0.shape)
                    jax.debug.print("=" * 60)
                jax.lax.cond(update_step == 0, _debug_print_env_step, lambda: None)

                env_act = {"agent_0": action_0.squeeze().flatten(), "agent_1": action_1.squeeze().flatten()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # shaped_reward 加算前に元報酬を保存
                original_reward_0 = reward["agent_0"]
                original_reward_1 = reward["agent_1"]

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                # info を agent ごとに分離
                info_0 = {
                    "shaped_reward": info["shaped_reward"]["agent_0"],
                    "original_reward": original_reward_0,
                    "anneal_factor": jnp.full((config["NUM_ENVS"],), anneal_factor),
                    "combined_reward": reward["agent_0"],
                }
                info_1 = {
                    "shaped_reward": info["shaped_reward"]["agent_1"],
                    "original_reward": original_reward_1,
                    "anneal_factor": jnp.full((config["NUM_ENVS"],), anneal_factor),
                    "combined_reward": reward["agent_1"],
                }

                transition_0 = Transition(
                    done["__all__"],
                    action_0.squeeze(),
                    value_0.squeeze(),
                    reward["agent_0"],
                    log_prob_0.squeeze(),
                    obs_0,
                    info_0,
                )
                transition_1 = Transition(
                    done["__all__"],
                    action_1.squeeze(),
                    value_1.squeeze(),
                    reward["agent_1"],
                    log_prob_1.squeeze(),
                    obs_1,
                    info_1,
                )

                # ===== 検証4: Transition の shape 確認（最初の1回だけ） =====
                def _debug_print_transition():
                    jax.debug.print("=" * 60)
                    jax.debug.print("[検証4] Transition shape 確認")
                    jax.debug.print("  transition_0.done shape: {s}", s=transition_0.done.shape)
                    jax.debug.print("  transition_0.action shape: {s}", s=transition_0.action.shape)
                    jax.debug.print("  transition_0.value shape: {s}", s=transition_0.value.shape)
                    jax.debug.print("  transition_0.reward shape: {s}", s=transition_0.reward.shape)
                    jax.debug.print("  transition_0.obs shape: {s}", s=transition_0.obs.shape)
                    jax.debug.print("  done['__all__'] shape: {s}", s=done["__all__"].shape)
                    jax.debug.print("=" * 60)
                jax.lax.cond(update_step == 0, _debug_print_transition, lambda: None)
                runner_state = (
                    train_state_0, train_state_1,
                    env_state, obsv,
                    done["__all__"],
                    update_step,
                    hstate_0, hstate_1,
                    rng,
                )
                return runner_state, (transition_0, transition_1)

            initial_hstate_0 = runner_state[-3]  # hstate_0
            initial_hstate_1 = runner_state[-2]  # hstate_1
            runner_state, (traj_batch_0, traj_batch_1) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE — agent ごとに独立
            (
                train_state_0, train_state_1,
                env_state, last_obs, last_done, update_step,
                hstate_0, hstate_1, rng,
            ) = runner_state

            # agent_0
            last_obs_0 = last_obs["agent_0"]
            ac_in_0 = (last_obs_0[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_0 = network_0.apply(train_state_0.params, hstate_0, ac_in_0)
            last_val_0 = last_val_0.squeeze()

            # agent_1
            last_obs_1 = last_obs["agent_1"]
            ac_in_1 = (last_obs_1[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_1 = network_1.apply(train_state_1.params, hstate_1, ac_in_1)
            last_val_1 = last_val_1.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages_0, targets_0 = _calculate_gae(traj_batch_0, last_val_0)
            advantages_1, targets_1 = _calculate_gae(traj_batch_1, last_val_1)

            # ===== 検証5: traj_batch・GAE・ミニバッチ の shape 確認（最初の1回だけ） =====
            def _debug_print_gae():
                jax.debug.print("=" * 60)
                jax.debug.print("[検証5] traj_batch・GAE shape 確認")
                jax.debug.print("  traj_batch_0.obs shape: {s}", s=traj_batch_0.obs.shape)
                jax.debug.print("  traj_batch_0.done shape: {s}", s=traj_batch_0.done.shape)
                jax.debug.print("  traj_batch_0.reward shape: {s}", s=traj_batch_0.reward.shape)
                jax.debug.print("  last_val_0 shape: {s}", s=last_val_0.shape)
                jax.debug.print("  advantages_0 shape: {s}", s=advantages_0.shape)
                jax.debug.print("  targets_0 shape: {s}", s=targets_0.shape)
                jax.debug.print("[検証6] ミニバッチ shape（計算値）")
                jax.debug.print("  permutation size: NUM_ENVS = {s}", s=config["NUM_ENVS"])
                jax.debug.print("  ミニバッチサイズ: NUM_ENVS/NUM_MINIBATCHES = {s}", s=config["NUM_ENVS"] // config["NUM_MINIBATCHES"])
                jax.debug.print("  ミニバッチ内データ数: {s} x NUM_STEPS({t})", s=config["NUM_ENVS"] // config["NUM_MINIBATCHES"], t=config["NUM_STEPS"])
                jax.debug.print("=" * 60)
            jax.lax.cond(update_step == 0, _debug_print_gae, lambda: None)

            # UPDATE NETWORK — agent ごとに独立して更新
            def _make_update_epoch(network):
                """agent ごとの _update_epoch を生成するファクトリ"""
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                            # RERUN NETWORK
                            _, pi, value = network.apply(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )

                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )

                            # CALCULATE ACTOR LOSS
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, init_hstate, traj_batch, advantages, targets, rng = (
                        update_state
                    )
                    rng, _rng = jax.random.split(rng)

                    init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ENVS"], -1))
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss
                return _update_epoch

            _update_epoch_0 = _make_update_epoch(network_0)
            _update_epoch_1 = _make_update_epoch(network_1)

            # 乱数を agent ごとに分ける
            rng, rng_update_0, rng_update_1 = jax.random.split(rng, 3)

            # agent_0 の更新
            update_state_0 = (
                train_state_0, initial_hstate_0,
                traj_batch_0, advantages_0, targets_0, rng_update_0,
            )
            update_state_0, loss_info_0 = jax.lax.scan(
                _update_epoch_0, update_state_0, None, config["UPDATE_EPOCHS"]
            )
            train_state_0 = update_state_0[0]

            # agent_1 の更新
            update_state_1 = (
                train_state_1, initial_hstate_1,
                traj_batch_1, advantages_1, targets_1, rng_update_1,
            )
            update_state_1, loss_info_1 = jax.lax.scan(
                _update_epoch_1, update_state_1, None, config["UPDATE_EPOCHS"]
            )
            train_state_1 = update_state_1[0]

            # wandb ログ — agent ごとにメトリクスを分けて記録
            metric_0 = traj_batch_0.info
            metric_1 = traj_batch_1.info

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric_0 = jax.tree_util.tree_map(lambda x: x.mean(), metric_0)
            metric_1 = jax.tree_util.tree_map(lambda x: x.mean(), metric_1)
            metric = {}
            for k, v in metric_0.items():
                metric[f"agent_0/{k}"] = v
            for k, v in metric_1.items():
                metric[f"agent_1/{k}"] = v
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state_0, train_state_1,
                env_state, last_obs, last_done,
                update_step,
                hstate_0, hstate_1,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_0, train_state_1,
            env_state, obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            0,
            init_hstate_0, init_hstate_1,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v2"
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2", "IndividualWeight"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_rnn_overcooked_v2_individual_{layout_name}",
    )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    # === 重み保存 ===
    from flax.serialization import to_bytes

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # runner_state: (ts_0, ts_1, env_state, obs, done, update_step, hs_0, hs_1, rng)
    # vmap で NUM_SEEDS 分あるので [0] で最初の seed を取り出す
    runner_state = out["runner_state"]
    params_0 = jax.tree_util.tree_map(lambda x: x[0], runner_state[0].params)
    params_1 = jax.tree_util.tree_map(lambda x: x[0], runner_state[1].params)

    save_path_0 = os.path.join(save_dir, f"params_{layout_name}_individual_agent0_{ts}.msgpack")
    save_path_1 = os.path.join(save_dir, f"params_{layout_name}_individual_agent1_{ts}.msgpack")
    with open(save_path_0, "wb") as f:
        f.write(to_bytes(params_0))
    with open(save_path_1, "wb") as f:
        f.write(to_bytes(params_1))
    print(f"Saved agent_0 weights to {save_path_0}")
    print(f"Saved agent_1 weights to {save_path_1}")


if __name__ == "__main__":
    main()
