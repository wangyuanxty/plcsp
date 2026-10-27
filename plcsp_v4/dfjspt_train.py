import json
import os
# from memory_profiler import profile
# @profile
# def func():
#     print("2")
import ray
from ray import air, tune, train
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.models import ModelCatalog
# from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
# from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
# from ray.rllib.algorithms.algorithm import Algorithm
from typing import Dict
# from gymnasium import spaces
# import torch
# import numpy as np

from plcsp_v4 import dfjspt_params
from plcsp_v4.dfjspt_env import DfjsptMaEnv
from plcsp_v4.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel


class MyCallbacks(DefaultCallbacks):

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.policy_config["batch_mode"] == "truncate_episodes":
        #     # Make sure this episode is really done.
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        episode.custom_metrics["total_makespan"] = episode.worker.env.final_makespan
        episode.custom_metrics["instance_id"] = episode.worker.env.current_instance_id
        episode.custom_metrics["instance_rule_makespan"] = episode.worker.env.rule_makespan_for_current_instance
        episode.custom_metrics["drl_minus_rule"] = episode.worker.env.drl_minus_rule


class MyTrainable(tune.Trainable):
    def setup(self, my_config):
        # self.max_iterations = 500
        self.config = PPOConfig().update_from_dict(my_config)
        self.agent1 = self.config.build()

        self.epoch = 0

    def step(self):
        result = self.agent1.train()
        if result["custom_metrics"]["total_makespan_mean"] <= result["custom_metrics"]["instance_rule_makespan_mean"] - 100:
            dfjspt_params.use_custom_loss = False
        if result["episodes_total"] >= 2e5:
            dfjspt_params.use_custom_loss = False
        self.epoch += 1
        return result

    def save_checkpoint(self, tmp_checkpoint_dir):
        self.agent1.save(tmp_checkpoint_dir)
        # print(f"Checkpoint saved in directory {tmp_checkpoint_dir}")
        return tmp_checkpoint_dir

    # def load_checkpoint(self, tmp_checkpoint_dir):
    #     checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
    #     self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":

    for _ in range(1):

        print(
            f"Start training with {dfjspt_params.n_jobs} jobs, {dfjspt_params.n_machines} machines, and {dfjspt_params.n_transbots} transbots.")

        log_dir = os.path.dirname(__file__) + "/training_results/J" + str(
            dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots)

        ModelCatalog.register_custom_model(
            "job_agent_model", JobActionMaskModel
        )
        ModelCatalog.register_custom_model(
            "machine_agent_model", MachineActionMaskModel
        )
        ModelCatalog.register_custom_model(
            "transbot_agent_model", TransbotActionMaskModel
        )

        example_env = DfjsptMaEnv({
            "train_or_eval_or_test": "train",
        })

        # Define the policies for each agent
        policies = {
            "policy_agent0": (
                None,
                # trained_job_policy,
                example_env.observation_space["agent0"],
                example_env.action_space["agent0"],
                {"model": {
                    "custom_model": "job_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent1": (
                None,
                # trained_machine_policy,
                example_env.observation_space["agent1"],
                example_env.action_space["agent1"],
                {"model": {
                    "custom_model": "machine_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent2": (
                None,
                # trained_transbot_policy,
                example_env.observation_space["agent2"],
                example_env.action_space["agent2"],
                {"model": {
                    "custom_model": "transbot_agent_model",
                    "fcnet_hiddens": [128, 128],
                        "fcnet_activation": "tanh",
                }}),
        }

        # Define the policy mapping function
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id == "agent0":
                return "policy_agent0"
            elif agent_id == "agent1":
                return "policy_agent1"
            else:
                return "policy_agent2"

        num_workers = dfjspt_params.num_workers
        num_gpu = dfjspt_params.num_gpu
        if num_gpu > 0:
            driver_gpu = 0.1
            worker_gpu = (1 - driver_gpu) / num_workers
        else:
            driver_gpu = 0
            worker_gpu = 0
        my_config = {
            # environment:
            # "env": 'CustomMeanStdObsEnv-v0',
            "env": DfjsptMaEnv,
            "env_config": {
                "train_or_eval_or_test": "train",
            },
            "disable_env_checking": True,
            # framework:
            "framework": dfjspt_params.framework,
            # rollouts:
            "num_rollout_workers": num_workers,
            "num_envs_per_worker": dfjspt_params.num_envs_per_worker,
            "batch_mode": "complete_episodes",
            # debugging：
            "log_level": "WARN",
            "log_sys_usage": True,
            # callbacks：
            "callbacks_class": MyCallbacks,
            # resources：
            "num_gpus": driver_gpu,
            "num_gpus_per_worker": worker_gpu,
            "num_cpus_per_worker": 0.5,  # [v4] 50 workers on a 32-core box
            "num_cpus_for_local_worker": 1,
            # evaluation:
            "evaluation_interval": 5,
            "evaluation_duration": 10,
            "evaluation_duration_unit": "episodes",
            "evaluation_parallel_to_training": True,
            "enable_async_evaluation": True,
            "evaluation_num_workers": 1,
            "evaluation_config": PPOConfig.overrides(
                env_config={
                    "train_or_eval_or_test": "eval",
                },
                explore=False,
            ),
            # training:
            "lr_schedule": [
                [0, 3e-5],
                [dfjspt_params.n_jobs * dfjspt_params.n_machines * 5e6, 1e-5]],
            "train_batch_size": dfjspt_params.n_jobs * dfjspt_params.n_machines * max(num_workers, 1) * 120,
            "sgd_minibatch_size": dfjspt_params.n_jobs * dfjspt_params.n_machines * 120,
            "num_sgd_iter": 10,
            "entropy_coeff": 0.001,
            # multi_agent
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        }

        stop = {
            "training_iteration": dfjspt_params.stop_iters,
        }

        if not dfjspt_params.use_tune:
            # manual training with train loop using PPO and fixed learning rate
            # if args.run != "PPO":
            #     raise ValueError("Only support --run PPO with --no-tune.")
            print("Running manual train loop without Ray Tune.")

            config = PPOConfig().update_from_dict(my_config)
            algo = config.build()

            for i in range(dfjspt_params.stop_iters):
                result = algo.train()
                if result["custom_metrics"]["total_makespan_mean"] <= result["custom_metrics"][
                    "instance_rule_makespan_mean"]:
                    dfjspt_params.use_custom_loss = False
                if result["episodes_total"] >= 1e+5:
                    dfjspt_params.use_custom_loss = False
                if i % 5 == 0:
                    print(pretty_print(result))
                if i % 20 == 0:
                    checkpoint_dir = algo.save()
                    print(f"Checkpoint saved in directory {checkpoint_dir}")

            checkpoint_dir_end = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir_end}")
            algo.stop()
        else:
            # automated run with Tune and grid search and TensorBoard
            print("Training automatically with Ray Tune")

            resources = PPO.default_resource_request(my_config)
            tuner = tune.Tuner(
                tune.with_resources(MyTrainable, resources=resources),
                param_space=my_config,
                run_config=air.RunConfig(
                    stop=stop,
                    name=log_dir,
                    checkpoint_config=train.CheckpointConfig(checkpoint_frequency=100, checkpoint_at_end=True)  # [v4] every 100 iters,
                ),
            )
            results = tuner.fit()

            # Get the best result based on a particular metric.
            best_result = results.get_best_result(metric="custom_metrics/total_makespan_mean", mode="min")
            print(best_result)

            # Get the best checkpoint corresponding to the best result.
            best_checkpoint = best_result.checkpoint
            print(best_checkpoint)

        ray.shutdown()



