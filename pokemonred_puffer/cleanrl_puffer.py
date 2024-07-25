# import heapq
# import math
# from multiprocessing import Queue
# import os
# import pathlib
# import random
# import time
# from collections import deque
# from types import SimpleNamespace
# from typing import Any, Callable
# import uuid
# from collections import defaultdict
# from datetime import timedelta

# import numpy as np
# import pufferlib
# import pufferlib.emulation
# import pufferlib.frameworks.cleanrl
# import pufferlib.policy_pool
# import pufferlib.utils
# import pufferlib.vectorization
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from pokemonred_puffer.eval import make_pokemon_red_overlay
# from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE

# import logging
# # Configure logging
# logging.basicConfig(
#     filename="diagnostics.log",  # Name of the log file
#     filemode="a",  # Append to the file
#     format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
#     level=logging.INFO,  # Log level
# )




# @pufferlib.dataclass
# class Performance:
#     total_uptime = 0
#     total_updates = 0
#     total_agent_steps = 0
#     epoch_time = 0
#     epoch_sps = 0
#     eval_time = 0
#     eval_sps = 0
#     eval_memory = 0
#     eval_pytorch_memory = 0
#     env_time = 0
#     env_sps = 0
#     inference_time = 0
#     inference_sps = 0
#     train_time = 0
#     train_sps = 0
#     train_memory = 0
#     train_pytorch_memory = 0


# @pufferlib.dataclass
# class Losses:
#     policy_loss = 0
#     value_loss = 0
#     entropy = 0
#     old_approx_kl = 0
#     approx_kl = 0
#     clipfrac = 0
#     explained_variance = 0


# @pufferlib.dataclass
# class Charts:
#     global_step = 0
#     SPS = 0
#     learning_rate = 0


# def rollout(
#     env_creator,
#     env_kwargs,
#     agent_creator,
#     agent_kwargs,
#     model_path=None,
#     device="cuda",
#     verbose=True,
# ):
#     env = env_creator(**env_kwargs)
#     if model_path is None:
#         agent = agent_creator(env, **agent_kwargs)
#     else:
#         agent = torch.load(model_path, map_location=device)

#     terminal = truncated = True

#     while True:
#         if terminal or truncated:
#             if verbose:
#                 print("---  Reset  ---")

#             ob, info = env.reset()
#             state = None
#             step = 0
#             return_val = 0

#         ob = torch.tensor(ob, device=device).unsqueeze(0)
#         with torch.no_grad():
#             if hasattr(agent, "lstm"):
#                 action, _, _, _, state = agent.get_action_and_value(ob, state)
#             else:
#                 action, _, _, _ = agent.get_action_and_value(ob)

#         ob, reward, terminal, truncated, _ = env.step(action[0].item())
#         return_val += reward

#         chars = env.render()
#         print("\033c", end="")
#         print(chars)

#         if verbose:
#             print(f"Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}")

#         time.sleep(0.5)
#         step += 1


# def seed_everything(seed, torch_deterministic):
#     random.seed(seed)
#     np.random.seed(seed)
#     if seed is not None:
#         torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = torch_deterministic


# def unroll_nested_dict(d):
#     if not isinstance(d, dict):
#         return d

#     for k, v in d.items():
#         if isinstance(v, dict):
#             for k2, v2 in unroll_nested_dict(v):
#                 yield f"{k}/{k2}", v2
#         else:
#             yield k, v


# def print_dashboard(stats, init_performance, performance):
#     output = []
#     data = {**stats, **init_performance, **performance}

#     grouped_data = defaultdict(dict)

#     for k, v in data.items():
#         if k == "total_uptime":
#             v = timedelta(seconds=v)
#         if "memory" in k:
#             v = pufferlib.utils.format_bytes(v)
#         elif "time" in k:
#             try:
#                 v = f"{v:.2f} s"
#             except:  # noqa
#                 pass

#         first_word, *rest_words = k.split("_")
#         rest_words = " ".join(rest_words).title()

#         grouped_data[first_word][rest_words] = v

#     for main_key, sub_dict in grouped_data.items():
#         output.append(f"{main_key.title()}")
#         for sub_key, sub_value in sub_dict.items():
#             output.append(f"    {sub_key}: {sub_value}")

#     print("\033c", end="")
#     print("\n".join(output))
#     time.sleep(1 / 20)


# class CleanPuffeRL:
#     def __init__(
#         self,
#         config: SimpleNamespace | None = None,
#         exp_name: str | None = None,
#         track: bool = False,
#         # Agent
#         agent: nn.Module | None = None,
#         agent_creator: Callable[..., Any] | None = None,
#         agent_kwargs: dict = None,
#         # Environment
#         env_creator: Callable[..., Any] | None = None,
#         env_creator_kwargs: dict | None = None,
#         vectorization: ... = pufferlib.vectorization.Serial,
#         # Policy Pool options
#         policy_selector: Callable[
#             [list[Any], int], list[Any]
#         ] = pufferlib.policy_pool.random_selector,
#     ):
#         self.config = config
#         if self.config is None:
#             self.config = pufferlib.args.CleanPuffeRL()

#         self.exp_name = exp_name
#         if self.exp_name is None:
#             self.exp_name = str(uuid.uuid4())[:8]

#         self.wandb = None
#         if track:
#             import wandb

#             self.wandb = wandb

#         self.start_time = time.time()
#         seed_everything(config.seed, config.torch_deterministic)
#         self.total_updates = config.total_timesteps // config.batch_size
#         self.total_agent_steps = 0

#         self.device = config.device

#         # Ensure that data_dir is set
#         if not hasattr(config, "data_dir") or config.data_dir is None:
#             config.data_dir = "./data"

#         # Create environments, agent, and optimizer
#         init_profiler = pufferlib.utils.Profiler(memory=True)
#         with init_profiler:
#             self.pool = vectorization(
#                 env_creator,
#                 env_kwargs=env_creator_kwargs,
#                 num_envs=config.num_envs,
#                 envs_per_worker=config.envs_per_worker,
#                 envs_per_batch=config.envs_per_batch,
#                 env_pool=config.env_pool,
#                 mask_agents=True,
#             )

#         obs_shape = self.pool.single_observation_space.shape
#         atn_shape = self.pool.single_action_space.shape
#         self.num_agents = self.pool.agents_per_env
#         total_agents = self.num_agents * config.num_envs

#         self.agent = pufferlib.emulation.make_object(
#             agent, agent_creator, [self.pool.driver_env], agent_kwargs
#         )
#         self.env_send_queues: list[Queue] = env_creator_kwargs["async_config"]["send_queues"]
#         self.env_recv_queues: list[Queue] = env_creator_kwargs["async_config"]["recv_queues"]

#         # If data_dir is provided, load the resume state
#         resume_state = {}
#         path = pathlib.Path(config.data_dir) / self.exp_name
#         print(f"Looking for checkpoints in: {path}")

#         trainer_path = path / "trainer_state.pt"
#         print(f"Trainer path: {trainer_path}")

#         if trainer_path.exists():
#             print("Found trainer state, loading...")
#             resume_state = torch.load(trainer_path)

#             model_version = str(resume_state["update"]).zfill(6)
#             model_filename = f"model_{model_version}_state.pth"
#             model_path = path / model_filename
#             print(f"Model path: {model_path}")

#             if model_path.exists():
#                 self.agent.load_state_dict(torch.load(model_path, map_location=self.device))
#                 print(
#                     f'Resumed from update {resume_state["update"]} '
#                     f'with policy {resume_state["model_name"]}'
#                 )
#             else:
#                 print(f"Model checkpoint {model_path} not found. Starting fresh.")
#         else:
#             print(f"Trainer state checkpoint {trainer_path} not found. Starting fresh.")

#         self.global_step = resume_state.get("global_step", 0)
#         self.agent_step = resume_state.get("agent_step", 0)
#         self.update = resume_state.get("update", 0)
#         self.lr_update = resume_state.get("lr_update", 0)

#         self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
#         self.opt_state = resume_state.get("optimizer_state_dict", None)

#         if config.compile:
#             self.agent = torch.compile(self.agent, mode=config.compile_mode)
#             # TODO: Figure out how to compile the optimizer!
#             # self.calculate_loss = torch.compile(self.calculate_loss, mode=config.compile_mode)

#         if config.load_optimizer_state is True and self.opt_state is not None:
#             self.optimizer.load_state_dict(resume_state["optimizer_state_dict"])

#         # Create policy pool
#         pool_agents = self.num_agents * self.pool.envs_per_batch
#         self.policy_pool = pufferlib.policy_pool.PolicyPool(
#             self.agent,
#             pool_agents,
#             atn_shape,
#             self.device,
#             path,
#             self.config.pool_kernel,
#             policy_selector,
#         )

#         # Allocate Storage
#         storage_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
#         self.pool.async_reset(config.seed)
#         self.next_lstm_state = None
#         if hasattr(self.agent, "lstm"):
#             shape = (self.agent.lstm.num_layers, total_agents, self.agent.lstm.hidden_size)
#             self.next_lstm_state = (
#                 torch.zeros(shape, device=self.device),
#                 torch.zeros(shape, device=self.device),
#             )
#         self.obs = torch.zeros(config.batch_size + 1, *obs_shape, dtype=torch.uint8)
#         self.actions = torch.zeros(config.batch_size + 1, *atn_shape, dtype=int)
#         self.logprobs = torch.zeros(config.batch_size + 1)
#         self.rewards = torch.zeros(config.batch_size + 1)
#         self.dones = torch.zeros(config.batch_size + 1)
#         self.truncateds = torch.zeros(config.batch_size + 1)
#         self.values = torch.zeros(config.batch_size + 1)

#         self.obs_ary = np.asarray(self.obs, dtype=np.uint8)
#         self.actions_ary = np.asarray(self.actions)
#         self.logprobs_ary = np.asarray(self.logprobs)
#         self.rewards_ary = np.asarray(self.rewards)
#         self.dones_ary = np.asarray(self.dones)
#         self.truncateds_ary = np.asarray(self.truncateds)
#         self.values_ary = np.asarray(self.values)

#         storage_profiler.stop()

#         # "charts/actions": wandb.Histogram(b_actions.cpu().numpy()),
#         self.init_performance = pufferlib.namespace(
#             init_time=time.time() - self.start_time,
#             init_env_time=init_profiler.elapsed,
#             init_env_memory=init_profiler.memory,
#             tensor_memory=storage_profiler.memory,
#             tensor_pytorch_memory=storage_profiler.pytorch_memory,
#         )

#         self.sort_keys = []
#         self.learning_rate = (config.learning_rate,)
#         self.losses = Losses()
#         self.performance = Performance()

#         self.reward_buffer = deque(maxlen=1_000)
#         self.exploration_map_agg = np.zeros((config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32)
#         self.cut_exploration_map_agg = np.zeros(
#             (config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32
#         )
#         self.taught_cut = False

#         self.infos = {}
#         self.log = False

#     @pufferlib.utils.profile
#     def evaluate(self):
#         config = self.config
#         # TODO: Handle update on resume
#         if self.log and self.wandb is not None and self.performance.total_uptime > 0:
#             self.wandb.log(
#                 {
#                     "SPS": self.SPS,
#                     "global_step": self.global_step,
#                     "learning_rate": self.optimizer.param_groups[0]["lr"],
#                     **{f"losses/{k}": v for k, v in self.losses.items()},
#                     **{f"performance/{k}": v for k, v in self.performance.items()},
#                     **{f"stats/{k}": v for k, v in self.stats.items()},
#                     **{f"max_stats/{k}": v for k, v in self.max_stats.items()},
#                     **{
#                         f"skillrank/{policy}": elo
#                         for policy, elo in self.policy_pool.ranker.ratings.items()
#                     },
#                 },
#             )
#             self.log = False

#         # now for a tricky bit:
#         # if we have swarm_frequency, we will take the top swarm_keep_pct envs and evenly distribute
#         # their states to the bottom 90%.
#         # we do this here so the environment can remain "pure"
#         if (
#             hasattr(self.config, "swarm_frequency")
#             and hasattr(self.config, "swarm_keep_pct")
#             and self.update % self.config.swarm_frequency == 0
#             and "learner" in self.infos
#             and "reward/event" in self.infos["learner"]
#         ):
#             # collect the top swarm_keep_pct % of envs
#             largest = [
#                 x[0]
#                 for x in heapq.nlargest(
#                     math.ceil(self.config.num_envs * self.config.swarm_keep_pct),
#                     enumerate(self.infos["learner"]["reward/event"]),
#                     key=lambda x: x[1],
#                 )
#             ]
#             print("Migrating states:")
#             waiting_for = []
#             # Need a way not to reset the env id counter for the driver env
#             # Until then env ids are 1-indexed
#             for i in range(self.config.num_envs):
#                 if i not in largest:
#                     new_state = random.choice(largest)
#                     print(
#                         f'\t {i+1} -> {new_state+1}, event scores: {self.infos["learner"]["reward/event"][i]} -> {self.infos["learner"]["reward/event"][new_state]}'
#                     )
#                     self.env_recv_queues[i + 1].put(self.infos["learner"]["state"][new_state])
#                     waiting_for.append(i + 1)
#                     # Now copy the hidden state over
#                     # This may be a little slow, but so is this whole process
#                     self.next_lstm_state[0][:, i, :] = self.next_lstm_state[0][:, new_state, :]
#                     self.next_lstm_state[1][:, i, :] = self.next_lstm_state[1][:, new_state, :]
#             for i in waiting_for:
#                 self.env_send_queues[i].get()
#             print("State migration complete")

#         self.policy_pool.update_policies()
#         env_profiler = pufferlib.utils.Profiler()
#         inference_profiler = pufferlib.utils.Profiler()
#         eval_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True).start()
#         misc_profiler = pufferlib.utils.Profiler()

#         ptr = step = padded_steps_collected = agent_steps_collected = 0
#         while True:
#             step += 1
#             if ptr == config.batch_size + 1:
#                 break

#             with env_profiler:
#                 o, r, d, t, i, env_id, mask = self.pool.recv()

#             with misc_profiler:
#                 i = self.policy_pool.update_scores(i, "return")
#                 # TODO: Update this for policy pool
#                 for ii, ee in zip(i["learner"], env_id):
#                     ii["env_id"] = ee

#             with inference_profiler, torch.no_grad():
#                 o = torch.as_tensor(o).to(device=self.device, non_blocking=True)
#                 r = (
#                     torch.as_tensor(r, dtype=torch.float32)
#                     .to(device=self.device, non_blocking=True)
#                     .view(-1)
#                 )
#                 d = (
#                     torch.as_tensor(d, dtype=torch.float32)
#                     .to(device=self.device, non_blocking=True)
#                     .view(-1)
#                 )

#                 agent_steps_collected += sum(mask)
#                 padded_steps_collected += len(mask)

#                 # Multiple policies will not work with new envpool
#                 next_lstm_state = self.next_lstm_state
#                 if next_lstm_state is not None:
#                     next_lstm_state = (
#                         next_lstm_state[0][:, env_id],
#                         next_lstm_state[1][:, env_id],
#                     )

#                 actions, logprob, value, next_lstm_state = self.policy_pool.forwards(
#                     o, next_lstm_state
#                 )

#                 if next_lstm_state is not None:
#                     h, c = next_lstm_state
#                     self.next_lstm_state[0][:, env_id] = h
#                     self.next_lstm_state[1][:, env_id] = c

#                 value = value.flatten()

#             with misc_profiler:
#                 actions = actions.cpu().numpy()

#                 # Index alive mask with policy pool idxs...
#                 # TODO: Find a way to avoid having to do this
#                 learner_mask = torch.as_tensor(mask * self.policy_pool.mask)

#                 # Ensure indices do not exceed batch size
#                 indices = torch.where(learner_mask)[0][: config.batch_size - ptr + 1].numpy()
#                 end = ptr + len(indices)

#                 # Batch indexing
#                 self.obs_ary[ptr:end] = o.cpu().numpy()[indices]
#                 self.values_ary[ptr:end] = value.cpu().numpy()[indices]
#                 self.actions_ary[ptr:end] = actions[indices]
#                 self.logprobs_ary[ptr:end] = logprob.cpu().numpy()[indices]
#                 self.rewards_ary[ptr:end] = r.cpu().numpy()[indices]
#                 self.dones_ary[ptr:end] = d.cpu().numpy()[indices]
#                 self.sort_keys.extend([(env_id[i], step) for i in indices])

#                 # Update pointer
#                 ptr += len(indices)

#                 for policy_name, policy_i in i.items():
#                     for agent_i in policy_i:
#                         for name, dat in unroll_nested_dict(agent_i):
#                             if policy_name not in self.infos:
#                                 self.infos[policy_name] = {}
#                             if name not in self.infos[policy_name]:
#                                 self.infos[policy_name][name] = [
#                                     np.zeros_like(dat)
#                                 ] * self.config.num_envs
#                             self.infos[policy_name][name][agent_i["env_id"]] = dat
#                             # infos[policy_name][name].append(dat)
#             with env_profiler:
#                 self.pool.send(actions)

#         eval_profiler.stop()

#         # Now that we initialized the model, we can get the number of parameters
#         if self.global_step == 0 and self.config.verbose:
#             self.n_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
#             print(f"Model Size: {self.n_params//1000} K parameters")

#         self.total_agent_steps += padded_steps_collected
#         new_step = np.mean(self.infos["learner"]["stats/step"])

#         if new_step > self.global_step:
#             self.global_step = new_step
#             self.log = True
#         self.reward = torch.mean(self.rewards).float().item()
#         self.SPS = int(padded_steps_collected / eval_profiler.elapsed)

#         perf = self.performance
#         perf.total_uptime = int(time.time() - self.start_time)
#         perf.total_agent_steps = self.total_agent_steps
#         perf.env_time = env_profiler.elapsed
#         perf.env_sps = int(agent_steps_collected / env_profiler.elapsed)
#         perf.inference_time = inference_profiler.elapsed
#         perf.inference_sps = int(padded_steps_collected / inference_profiler.elapsed)
#         perf.eval_time = eval_profiler.elapsed
#         perf.eval_sps = int(padded_steps_collected / eval_profiler.elapsed)
#         perf.eval_memory = eval_profiler.end_mem
#         perf.eval_pytorch_memory = eval_profiler.end_torch_mem
#         perf.misc_time = misc_profiler.elapsed

#         self.stats = {}
#         self.max_stats = {}
#         for k, v in self.infos["learner"].items():
#             if "pokemon_exploration_map" in k and config.save_overlay is True:
#                 if self.update % config.overlay_interval == 0:
#                     overlay = make_pokemon_red_overlay(
#                         np.stack(v, axis=0)
#                     )
#                     if self.wandb is not None:
#                         self.stats["Media/aggregate_exploration_map"] = self.wandb.Image(overlay)
#             elif "state" in k:
#                 continue
#             else:
#                 try:  # TODO: Better checks on log data types
#                     self.stats[k] = np.mean(v)
#                     self.max_stats[k] = np.max(v)
#                 except:  # noqa
#                     continue


#         if config.verbose:
#             print_dashboard(self.stats, self.init_performance, self.performance)

#         return self.stats, self.infos

#     @pufferlib.utils.profile
#     def train(self):
#         if self.done_training():
#             raise RuntimeError(f"Max training updates {self.total_updates} already reached")

#         config = self.config
#         # assert data.num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"

#         train_profiler = pufferlib.utils.Profiler(memory=True, pytorch_memory=True)
#         train_profiler.start()

#         if config.anneal_lr:
#             frac = 1.0 - (self.lr_update - 1.0) / self.total_updates
#             lrnow = frac * config.learning_rate
#             self.optimizer.param_groups[0]["lr"] = lrnow

#         num_minibatches = config.batch_size // config.bptt_horizon // config.batch_rows
#         assert (
#             num_minibatches > 0
#         ), "config.batch_size // config.bptt_horizon // config.batch_rows must be > 0"
#         idxs = sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__)
#         self.sort_keys = []
#         b_idxs = (
#             torch.tensor(idxs, dtype=torch.long)[:-1]
#             .reshape(config.batch_rows, num_minibatches, config.bptt_horizon)
#             .transpose(0, 1)
#         )

#         # bootstrap value if not done
#         with torch.no_grad():
#             advantages = torch.zeros(config.batch_size, device=self.device)
#             lastgaelam = 0
#             try:
#                 for t in reversed(range(config.batch_size)):
#                     i, i_nxt = idxs[t], idxs[t + 1]
#                     nextnonterminal = 1.0 - self.dones[i_nxt]
#                     nextvalues = self.values[i_nxt]
#                     delta = (
#                         self.rewards[i_nxt]
#                         + config.gamma * nextvalues * nextnonterminal
#                         - self.values[i]
#                     )
#                     advantages[t] = lastgaelam = (
#                         delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
#                     )
#             except Exception as e:
#                 logging.info(f"Error in cleanrl_puffer.py: {e}")

#         # Flatten the batch
#         self.b_obs = b_obs = torch.as_tensor(self.obs_ary[b_idxs], dtype=torch.uint8)
#         b_actions = torch.as_tensor(self.actions_ary[b_idxs]).to(self.device, non_blocking=True)
#         b_logprobs = torch.as_tensor(self.logprobs_ary[b_idxs]).to(self.device, non_blocking=True)
#         # b_dones = torch.as_tensor(self.dones_ary[b_idxs]).to(self.device, non_blocking=True)
#         b_values = torch.as_tensor(self.values_ary[b_idxs]).to(self.device, non_blocking=True)
#         b_advantages = advantages.reshape(
#             config.batch_rows, num_minibatches, config.bptt_horizon
#         ).transpose(0, 1)
#         b_returns = b_advantages + b_values

#         # Optimizing the policy and value network
#         train_time = time.time()
#         pg_losses, entropy_losses, v_losses, clipfracs, old_kls, kls = [], [], [], [], [], []
#         mb_obs_buffer = torch.zeros_like(
#             b_obs[0], pin_memory=(self.device == "cuda"), dtype=torch.uint8
#         )

#         for epoch in range(config.update_epochs):
#             lstm_state = None
#             for mb in range(num_minibatches):
#                 mb_obs_buffer.copy_(b_obs[mb], non_blocking=True)
#                 mb_obs = mb_obs_buffer.to(self.device, non_blocking=True)
#                 mb_actions = b_actions[mb].contiguous()
#                 mb_values = b_values[mb].reshape(-1)
#                 mb_advantages = b_advantages[mb].reshape(-1)
#                 mb_returns = b_returns[mb].reshape(-1)

#                 if hasattr(self.agent, "lstm"):
#                     (
#                         _,
#                         newlogprob,
#                         entropy,
#                         newvalue,
#                         lstm_state,
#                     ) = self.agent.get_action_and_value(mb_obs, state=lstm_state, action=mb_actions)
#                     lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
#                 else:
#                     _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
#                         mb_obs.reshape(-1, *self.pool.single_observation_space.shape),
#                         action=mb_actions,
#                     )

#                 logratio = newlogprob - b_logprobs[mb].reshape(-1)
#                 ratio = logratio.exp()

#                 with torch.no_grad():
#                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
#                     old_approx_kl = (-logratio).mean()
#                     old_kls.append(old_approx_kl.item())
#                     approx_kl = ((ratio - 1) - logratio).mean()
#                     kls.append(approx_kl.item())
#                     clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

#                 mb_advantages = mb_advantages.reshape(-1)
#                 if config.norm_adv:
#                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (
#                         mb_advantages.std() + 1e-8
#                     )

#                 # Policy loss
#                 pg_loss1 = -mb_advantages * ratio
#                 pg_loss2 = -mb_advantages * torch.clamp(
#                     ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
#                 )
#                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()
#                 pg_losses.append(pg_loss.item())

#                 # Value loss
#                 newvalue = newvalue.view(-1)
#                 if self.config.clip_vloss:
#                     v_loss_unclipped = (newvalue - mb_returns) ** 2
#                     v_clipped = mb_values + torch.clamp(
#                         newvalue - mb_values,
#                         -self.config.vf_clip_coef,
#                         self.config.vf_clip_coef,
#                     )
#                     v_loss_clipped = (v_clipped - mb_returns) ** 2
#                     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
#                     v_loss = 0.5 * v_loss_max.mean()
#                 else:
#                     v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
#                 v_losses.append(v_loss.item())

#                 entropy_loss = entropy.mean()
#                 entropy_losses.append(entropy_loss.item())

#                 self.calculate_loss(pg_loss, entropy_loss, v_loss)

#             if config.target_kl is not None:
#                 if approx_kl > config.target_kl:
#                     break

#         train_profiler.stop()
#         y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
#         var_y = np.var(y_true)
#         explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

#         losses = self.losses
#         losses.policy_loss = np.mean(pg_losses)
#         losses.value_loss = np.mean(v_losses)
#         losses.entropy = np.mean(entropy_losses)
#         losses.old_approx_kl = np.mean(old_kls)
#         losses.approx_kl = np.mean(kls)
#         losses.clipfrac = np.mean(clipfracs)
#         losses.explained_variance = explained_var

#         perf = self.performance
#         perf.total_uptime = int(time.time() - self.start_time)
#         perf.total_updates = self.update + 1
#         perf.train_time = time.time() - train_time
#         perf.train_sps = int(config.batch_size / perf.train_time)
#         perf.train_memory = train_profiler.end_mem
#         perf.train_pytorch_memory = train_profiler.end_torch_mem
#         perf.epoch_time = perf.eval_time + perf.train_time
#         perf.epoch_sps = int(config.batch_size / perf.epoch_time)

#         if config.verbose:
#             print_dashboard(self.stats, self.init_performance, self.performance)

#         self.update += 1
#         self.lr_update += 1

#         if self.update % config.checkpoint_interval == 0 or self.done_training():
#             self.save_checkpoint()

#     def close(self):
#         self.pool.close()

#         if self.wandb is not None:
#             artifact_name = f"{self.exp_name}_model"
#             artifact = self.wandb.Artifact(artifact_name, type="model")
#             model_path = self.save_checkpoint()
#             artifact.add_file(model_path)
#             self.wandb.run.log_artifact(artifact)
#             self.wandb.finish()

#     def save_checkpoint(self):
#         if self.config.save_checkpoint is False:
#             return

#         path = os.path.join(self.config.data_dir, self.exp_name)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         model_name = f"model_{self.update:06d}_state.pth"
#         model_path = os.path.join(path, model_name)

#         # Already saved
#         if os.path.exists(model_path):
#             return model_path

#         # To handleboth uncompiled and compiled self.agent, when getting state_dict()
#         torch.save(getattr(self.agent, "_orig_mod", self.agent).state_dict(), model_path)

#         state = {
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "global_step": self.global_step,
#             "agent_step": self.global_step,
#             "update": self.update,
#             "model_name": model_name,
#         }

#         if self.wandb:
#             state["exp_name"] = self.exp_name

#         state_path = os.path.join(path, "trainer_state.pt")
#         torch.save(state, state_path + ".tmp")
#         os.rename(state_path + ".tmp", state_path)

#         # Also save a copy
#         torch.save(state, os.path.join(path, f"trainer_state_{self.update:06d}.pt"))

#         print(f"Model saved to {model_path}")

#         return model_path

#     def calculate_loss(self, pg_loss, entropy_loss, v_loss):
#         loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
#         self.optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
#         self.optimizer.step()

#     def done_training(self):
#         return self.update >= self.total_updates

#     def __enter__(self):
#         return self

#     def __exit__(self, *args):
#         print("Done training.")
#         self.save_checkpoint()
#         self.close()
#         print("Run complete")


import argparse
import heapq
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import Queue

import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.pytorch
import pufferlib.utils
import pufferlib.vector

# Fast Cython GAE implementation
import pyximport
import rich
import torch
from torch import nn
from rich.console import Console
from rich.table import Table

import wandb
from pokemonred_puffer.eval import make_pokemon_red_overlay
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE
from pokemonred_puffer.profile import Profile, Utilization

pyximport.install(setup_args={"include_dirs": np.get_include()})
from pokemonred_puffer.c_gae import compute_gae  # type: ignore  # noqa: E402


def rollout(
    env_creator,
    env_kwargs,
    agent_creator,
    agent_kwargs,
    model_path=None,
    device="cuda",
):
    # We are just using Serial vecenv to give a consistent
    # single-agent/multi-agent API for evaluation
    try:
        env = pufferlib.vector.make(
            env_creator, env_kwargs={"render_mode": "rgb_array", **env_kwargs}
        )
    except:  # noqa: E722
        env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs)

    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    ob, info = env.reset()
    driver = env.driver_env
    os.system("clear")
    state = None

    while True:
        render = driver.render()
        if driver.render_mode == "ansi":
            print("\033[0;0H" + render + "\n")
            time.sleep(0.6)
        elif driver.render_mode == "rgb_array":
            import cv2

            render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", render)
            cv2.waitKey(1)
            time.sleep(1 / 24)

        with torch.no_grad():
            ob = torch.from_numpy(ob).to(device)
            if hasattr(agent, "lstm"):
                action, _, _, _, state = agent(ob, state)
            else:
                action, _, _, _ = agent(ob)

            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward = env.step(action)[:2]
        reward = reward.mean()
        print(f"Reward: {reward:.4f}")


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v


# def count_params(policy: nn.Module):
#     return sum(p.numel() for p in policy.parameters() if p.requires_grad)

def count_params(policy: nn.Module, observation_space: torch.Size, device: torch.device):
    dummy_obs = torch.zeros(1, *observation_space).to(device)
    with torch.no_grad():
        if hasattr(policy, 'lstm'):
            lstm_state = (
                torch.zeros(policy.lstm.num_layers, 1, policy.lstm.hidden_size).to(device),
                torch.zeros(policy.lstm.num_layers, 1, policy.lstm.hidden_size).to(device)
            )
            policy(dummy_obs, lstm_state)
        else:
            policy(dummy_obs)
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)

@dataclass
class Losses:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    old_approx_kl: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    explained_variance: float = 0.0


@dataclass
class CleanPuffeRL:
    exp_name: str
    config: argparse.Namespace
    vecenv: pufferlib.vector.Serial | pufferlib.vector.Multiprocessing
    policy: nn.Module
    env_send_queues: list[Queue]
    env_recv_queues: list[Queue]
    wandb_client: wandb.wandb_sdk.wandb_run.Run | None = None
    profile: Profile = field(default_factory=lambda: Profile())
    losses: Losses = field(default_factory=lambda: Losses())
    global_step: int = 0
    epoch: int = 0
    stats: dict = field(default_factory=lambda: {})
    msg: str = ""
    infos: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        seed_everything(self.config.seed, self.config.torch_deterministic)
        if self.config.verbose:
            self.utilization = Utilization()
            print_dashboard(
                self.config.env,
                self.utilization,
                0,
                0,
                self.profile,
                self.losses,
                {},
                self.msg,
                clear=True,
            )

        self.vecenv.async_reset(self.config.seed)
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        total_agents = self.vecenv.num_agents

        self.lstm = self.policy.lstm if hasattr(self.policy, "lstm") else None
        self.experience = Experience(
            self.config.batch_size,
            self.vecenv.agents_per_batch,
            self.config.bptt_horizon,
            self.config.minibatch_size,
            obs_shape,
            obs_dtype,
            atn_shape,
            self.config.cpu_offload,
            self.config.device,
            self.lstm,
            total_agents,
        )

        self.uncompiled_policy = self.policy

        if self.config.compile:
            self.policy = torch.compile(self.policy, mode=self.config.compile_mode)
            


        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        self.last_log_time = time.time()

        self.reward_buffer = deque(maxlen=1_000)
        self.exploration_map_agg = np.zeros(
            (self.config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32
        )
        self.cut_exploration_map_agg = np.zeros(
            (self.config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32
        )
        self.taught_cut = False
        self.log = False

        # dummy batch to init lazy modules
        # After loading or creating your model, add a dummy forward pass to initialize the parameters
        dummy_obs = torch.zeros(1, *self.vecenv.single_observation_space.shape).to(self.config.device)
        if self.lstm:
            # If your model has an LSTM, initialize the LSTM state with the correct dimensions
            dummy_state = (
                torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size).to(self.config.device),
                torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size).to(self.config.device)
            )
            with torch.no_grad():
                self.policy(dummy_obs, dummy_state)
        else:
            with torch.no_grad():
                self.policy(dummy_obs)


    @pufferlib.utils.profile
    def evaluate(self):
        
        # Clear all self.infos except for the state
        for k in list(self.infos.keys()):
            if k != "state":
                del self.infos[k]

        # now for a tricky bit:
        # if we have swarm_frequency, we will take the top swarm_keep_pct envs and evenly distribute
        # their states to the bottom 90%.
        # we do this here so the environment can remain "pure"
        if (
            self.config.async_wrapper
            and hasattr(self.config, "swarm_frequency")
            and hasattr(self.config, "swarm_keep_pct")
            and self.epoch % self.config.swarm_frequency == 0
            and "reward/event" in self.infos
            and "state" in self.infos
        ):
            # collect the top swarm_keep_pct % of envs
            largest = [
                x[0]
                for x in heapq.nlargest(
                    math.ceil(self.config.num_envs * self.config.swarm_keep_pct),
                    enumerate(self.infos["reward/event"]),
                    key=lambda x: x[1],
                )
            ]
            print("Migrating states:")
            waiting_for = []
            # Need a way not to reset the env id counter for the driver env
            # Until then env ids are 1-indexed
            for i in range(self.config.num_envs):
                if i not in largest:
                    new_state = random.choice(largest)
                    print(
                        f'\t {i+1} -> {new_state+1}, event scores: {self.infos["reward/event"][i]} -> {self.infos["reward/event"][new_state]}'
                    )
                    self.env_recv_queues[i + 1].put(self.infos["state"][new_state])
                    waiting_for.append(i + 1)
                    # Now copy the hidden state over
                    # This may be a little slow, but so is this whole process
                    self.next_lstm_state[0][:, i, :] = self.next_lstm_state[0][:, new_state, :]
                    self.next_lstm_state[1][:, i, :] = self.next_lstm_state[1][:, new_state, :]
            for i in waiting_for:
                self.env_send_queues[i].get()
            print("State migration complete")

        with self.profile.eval_misc:
            policy = self.policy
            lstm_h, lstm_c = self.experience.lstm_h, self.experience.lstm_c
            
        # Initialize dummy state
        dummy_ob = torch.zeros((1, *self.vecenv.single_observation_space.shape), device=self.config.device)
        with torch.no_grad():
            if lstm_h is not None:
                self.policy(dummy_ob, (lstm_h[:, :1, :], lstm_c[:, :1, :]))
            else:
                self.policy(dummy_ob)

        while not self.experience.full:
            with self.profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                env_id = env_id.tolist()

            with self.profile.eval_misc:
                self.global_step += sum(mask)

                o = torch.as_tensor(o)
                o_device = o.to(self.config.device)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with self.profile.eval_forward, torch.no_grad():
                # TODO: In place-update should be faster. Leaking 7% speed max
                # Also should be using a cuda tensor to index
                if lstm_h is not None:
                    h = lstm_h[:, env_id]
                    c = lstm_c[:, env_id]
                    actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                    lstm_h[:, env_id] = h
                    lstm_c[:, env_id] = c
                else:
                    actions, logprob, _, value = policy(o_device)

                if self.config.device == "cuda":
                    torch.cuda.synchronize()

            with self.profile.eval_misc:
                value = value.flatten()
                actions = actions.cpu().numpy()
                mask = torch.as_tensor(mask)  # * policy.mask)
                o = o if self.config.cpu_offload else o_device
                if self.config.num_envs == 1:
                    actions = np.expand_dims(actions, 0)
                    logprob = logprob.unsqueeze(0)
                self.experience.store(o, value, actions, logprob, r, d, env_id, mask)

                for i in info:
                    for k, v in pufferlib.utils.unroll_nested_dict(i):
                        if k == "state":
                            self.infos[k] = [v]
                        else:
                            self.infos[k].append(v)

            with self.profile.env:
                self.vecenv.send(actions)

        with self.profile.eval_misc:
            self.stats = {}

            for k, v in self.infos.items():
                # Moves into models... maybe. Definitely moves.
                # You could also just return infos and have it in demo
                if "pokemon_exploration_map" in k and self.config.save_overlay is True:
                    if self.epoch % self.config.overlay_interval == 0:
                        overlay = make_pokemon_red_overlay(np.stack(self.infos[k], axis=0))
                        if self.wandb_client is not None:
                            self.stats["Media/aggregate_exploration_map"] = wandb.Image(
                                overlay) # , file_type="jpg"
                            # )
                elif "state" in k:
                    continue

                try:  # TODO: Better checks on log data types
                    self.stats[k] = np.mean(v)
                except:  # noqa: E722
                    continue

            if self.config.verbose:
                self.msg = f"Model Size: {abbreviate(count_params(self.policy, self.vecenv.single_observation_space.shape, self.config.device))} parameters"
                print_dashboard(
                    self.config.env,
                    self.utilization,
                    self.global_step,
                    self.epoch,
                    self.profile,
                    self.losses,
                    self.stats,
                    self.msg,
                )

        return self.stats, self.infos

    @pufferlib.utils.profile
    def train(self):
        self.losses = Losses()
        losses = self.losses

        with self.profile.train_misc:
            idxs = self.experience.sort_training_data()
            dones_np = self.experience.dones_np[idxs]
            values_np = self.experience.values_np[idxs]
            rewards_np = self.experience.rewards_np[idxs]
            # TODO: bootstrap between segment bounds
            advantages_np = compute_gae(
                dones_np, values_np, rewards_np, self.config.gamma, self.config.gae_lambda
            )
            self.experience.flatten_batch(advantages_np)

        for _ in range(self.config.update_epochs):
            lstm_state = None
            for mb in range(self.experience.num_minibatches):
                with self.profile.train_misc:
                    obs = self.experience.b_obs[mb]
                    obs = obs.to(self.config.device)
                    atn = self.experience.b_actions[mb]
                    log_probs = self.experience.b_logprobs[mb]
                    val = self.experience.b_values[mb]
                    adv = self.experience.b_advantages[mb]
                    ret = self.experience.b_returns[mb]

                with self.profile.train_forward:
                    if self.experience.lstm_h is not None:
                        _, newlogprob, entropy, newvalue, lstm_state = self.policy(
                            obs, state=lstm_state, action=atn
                        )
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = self.policy(
                            obs.reshape(-1, *self.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

                with self.profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if self.config.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -self.config.vf_clip_coef,
                            self.config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                    )

                with self.profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

                with self.profile.train_misc:
                    losses.policy_loss += pg_loss.item() / self.experience.num_minibatches
                    losses.value_loss += v_loss.item() / self.experience.num_minibatches
                    losses.entropy += entropy_loss.item() / self.experience.num_minibatches
                    losses.old_approx_kl += old_approx_kl.item() / self.experience.num_minibatches
                    losses.approx_kl += approx_kl.item() / self.experience.num_minibatches
                    losses.clipfrac += clipfrac.item() / self.experience.num_minibatches

            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break

        with self.profile.train_misc:
            if self.config.anneal_lr:
                frac = 1.0 - self.global_step / self.config.total_timesteps
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = self.experience.values_np
            y_true = self.experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            losses.explained_variance = explained_var
            self.epoch += 1

            done_training = self.global_step >= self.config.total_timesteps
            if self.profile.update(self) or done_training:
                if self.config.verbose:
                    print_dashboard(
                        self.config.env,
                        self.utilization,
                        self.global_step,
                        self.epoch,
                        self.profile,
                        self.losses,
                        self.stats,
                        self.msg,
                    )

                if (
                    self.wandb_client is not None
                    and self.global_step > 0
                    and time.time() - self.last_log_time > 5.0
                ):
                    self.last_log_time = time.time()
                    self.wandb_client.log(
                        {
                            "Overview/SPS": self.profile.SPS,
                            "Overview/agent_steps": self.global_step,
                            "Overview/learning_rate": self.optimizer.param_groups[0]["lr"],
                            **{f"environment/{k}": v for k, v in self.stats.items()},
                            **{f"losses/{k}": v for k, v in self.losses.__dict__.items()},
                            **{f"performance/{k}": v for k, v in self.profile},
                        }
                    )

            if self.epoch % self.config.checkpoint_interval == 0 or done_training:
                self.save_checkpoint()
                self.msg = f"Checkpoint saved at update {self.epoch}"

    def close(self):
        self.vecenv.close()
        if self.config.verbose:
            self.utilization.stop()

        if self.wandb_client is not None:
            artifact_name = f"{self.exp_name}_model"
            artifact = wandb.Artifact(artifact_name, type="model")
            model_path = self.save_checkpoint()
            artifact.add_file(model_path)
            self.wandb_client.log_artifact(artifact)
            self.wandb_client.finish()

    def save_checkpoint(self):
        config = self.config
        path = os.path.join(config.data_dir, config.exp_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy, model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "exp_id": config.exp_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        return model_path

    def calculate_loss(self, pg_loss, entropy_loss, v_loss):
        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

    def done_training(self):
        return self.global_step >= self.config.total_timesteps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Done training.")
        self.save_checkpoint()
        self.close()
        print("Run complete")


class Experience:
    """Flat tensor storage and array views for faster indexing"""

    def __init__(
        self,
        batch_size: int,
        agents_per_batch: int,
        bptt_horizon: int,
        minibatch_size: int,
        obs_shape: tuple[int],
        obs_dtype: np.dtype,
        atn_shape: tuple[int],
        cpu_offload: bool = False,
        device: str = "cuda",
        lstm: torch.nn.LSTM | None = None,
        lstm_total_agents: int = 0,
    ):
        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        pin = device == "cuda" and cpu_offload
        # obs_device = device if not pin else "cpu"
        self.obs = torch.zeros(
            batch_size,
            *obs_shape,
            dtype=obs_dtype,
            pin_memory=pin,
            device=device if not pin else "cpu",
        )
        self.actions = torch.zeros(batch_size, *atn_shape, dtype=int, pin_memory=pin)
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)

        # self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError("batch_size must be divisible by minibatch_size")

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError("minibatch_size must be divisible by bptt_horizon")

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(
        self,
        obs: torch.Tensor,
        value: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        env_id: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(
                    self.minibatch_rows, self.num_minibatches, self.bptt_horizon
                ).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        self.ptr = 0
        self.step = 0
        return idxs

    def flatten_batch(self, advantages_np: np.ndarray):
        advantages = torch.from_numpy(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = (
            advantages.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon)
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values


ROUND_OPEN = rich.box.Box(
    "╭──╮\n"  # noqa: F401
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = "[bright_cyan]"
c2 = "[white]"
c3 = "[cyan]"
b1 = "[bright_cyan]"
b2 = "[bright_white]"


def abbreviate(num):
    if num < 1e3:
        return f"{b2}{num:.0f}"
    elif num < 1e6:
        return f"{b2}{num/1e3:.1f}{c2}k"
    elif num < 1e9:
        return f"{b2}{num/1e6:.1f}{c2}m"
    elif num < 1e12:
        return f"{b2}{num/1e9:.1f}{c2}b"
    else:
        return f"{b2}{num/1e12:.1f}{c2}t"


def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return (
        f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s"
        if h
        else f"{b2}{m}{c2}m {b2}{s}{c2}s"
        if m
        else f"{b2}{s}{c2}s"
    )


def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100 * time / uptime - 1e-5)
    return f"{c1}{name}", duration(time), f"{b2}{percent:2d}%"


# TODO: Add env name to print_dashboard
def print_dashboard(
    env_name: str,
    utilization: Utilization,
    global_step: int,
    epoch: int,
    profile: Profile,
    losses: Losses,
    stats,
    msg: str,
    clear: bool = False,
    max_stats=None,
):
    if not max_stats:
        max_stats = [0]
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True, show_header=False, border_style="bright_cyan")

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = np.mean(utilization.cpu_util)
    dram_percent = np.mean(utilization.cpu_mem)
    gpu_percent = np.mean(utilization.gpu_util)
    vram_percent = np.mean(utilization.gpu_mem)
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f":blowfish: {c1}PufferLib {b2}1.0.0",
        f"{c1}CPU: {c3}{cpu_percent:.1f}%",
        f"{c1}GPU: {c3}{gpu_percent:.1f}%",
        f"{c1}DRAM: {c3}{dram_percent:.1f}%",
        f"{c1}VRAM: {c3}{vram_percent:.1f}%",
    )

    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify="left", vertical="top", width=16)
    s.add_column(f"{c1}Value", justify="right", vertical="top", width=8)
    s.add_row(f"{c2}Environment", f"{b2}{env_name}")
    s.add_row(f"{c2}Agent Steps", abbreviate(global_step))
    s.add_row(f"{c2}SPS", abbreviate(profile.SPS))
    s.add_row(f"{c2}Epoch", abbreviate(epoch))
    s.add_row(f"{c2}Uptime", duration(profile.uptime))
    s.add_row(f"{c2}Remaining", duration(profile.remaining))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf("Evaluate", profile.eval_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.eval_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Env", profile.env_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf("Train", profile.train_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Learn", profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.train_misc_time, profile.uptime))

    l = Table(  # noqa: E741
        box=None,
        expand=True,
    )
    l.add_column(f"{c1}Losses", justify="left", width=16)
    l.add_column(f"{c1}Value", justify="right", width=8)
    for metric, value in losses.__dict__.items():
        l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)
    i = 0
    for metric, value in stats.items():
        try:  # Discard non-numeric values
            int(value)
        except:  # noqa: E722
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
        i += 1

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row("", "")

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f" {c1}Message: {c2}{msg}")

    with console.capture() as capture:
        console.print(dashboard)

    print("\033[0;0H" + capture.get())
