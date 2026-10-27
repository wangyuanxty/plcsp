from gymnasium.spaces import Dict
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from plcsp_v4 import dfjspt_params
from plcsp_v4.dfjspt_generate_a_sample_batch import generate_sample_batch

torch, nn = try_import_torch()


class JobActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="job")

            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)

            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')


class MachineActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="machine")

            # Define a secondary loss by building a graph copy with weight sharing.
            # obs = batch["obs"]
            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        else:
            return policy_loss


class TransbotActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observation" in orig_space.spaces
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Get the next batch from our input files.
            batch = generate_sample_batch(batch_type="transbot")

            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.obs_space,
                tensorlib="torch",
            )
            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = obs["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist = TorchCategorical(logits, self.model_config)

            imitation_loss = torch.mean(
                -action_dist.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')