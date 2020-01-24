import abc
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard


class Buddy:
    def __init__(self, experiment_name, model, optimizer_names=["primary"], load_checkpoint=True,
                 log_dir="logs", checkpoint_dir="checkpoints"):
        """
        Buddy is a model manager that abstracts away PyTorch boilerplate.

        Helps with:
            - Creating/using/managing optimizers
            - Checkpointing (models + optimizers)
            - Namespaced/scoped Tensorboard logging
        """
        # CUDA boilerplate
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        print("Using device:", self._device)
        torch.autograd.set_detect_anomaly(True)

        # Model and experiment parameters
        assert isinstance(model, nn.Module)
        self._experiment_name = experiment_name
        self._model = model

        # State variables for tensorboard
        # The writer is lazily instantiated in TrainingBuddy.log()
        self._writer = None
        self._log_dir = log_dir
        self._log_scopes = []

        # Checkpointing variables
        self._checkpoint_dir = checkpoint_dir
        self._steps = 0

        # Create optimizers -- we use a different one for each loss function
        # TODO: add support for non-Adadelta optimizers (at least ADAM...)
        self._optimizers = {}
        for optimizer_name in optimizer_names:
            self._optimizers[optimizer_name] = optim.Adadelta(
                self._model.parameters())

        # Load checkpoint using model name
        if load_checkpoint:
            self.load_checkpoint()

    def minimize(self, loss, retain_graph=False,
                 optimizer_name="primary", checkpoint_interval=1000):
        """
        Compute gradients and use them to minimize a loss function.
        """

        assert optimizer_name in self._optimizers.keys()

        # Take gradient step
        self._optimizers[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizers[optimizer_name].step()

        # Update step & checkpoint
        self._steps += 1
        if self._steps % checkpoint_interval == 0:
            self.save_checkpoint()

    def log_scope(self, scope):
        """
        Returns a scope to log tensors in.

        Example usage:

        ```
            with buddy.log_scope("scope"):
                # Logs to scope/loss
                buddy.log("loss", loss_tensor)
        ```
        """
        class _Namespace:
            def __enter__(unused_self):
                self._log_scopes.append(scope)
                return unused_self

            def __exit__(*unused):
                self._log_scopes.pop()
                return

        return _Namespace()

    def log(self, name, value):
        """
        Log a tensor for visualization in Tensorboard. Currently only supports scalars.
        """
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(
                self._log_dir + "/" + self._experiment_name)

        self._writer.add_scalar(name, value, global_step=self._steps)

    def save_checkpoint(self, label=None, path=None):
        """
        Saves a checkpoint!
        """

        # Create directory if it doesn't exist yet
        if not os.path.isdir(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        if path is None and label is None:
            path = "{}/{}-{:016d}.ckpt".format(self._checkpoint_dir,
                                               self._experiment_name, self._steps)
        else:
            path = "{}/{}-{}.ckpt".format(self._checkpoint_dir,
                                          self._experiment_name, label)

        optimizer_states = {}
        for name, optimizer in self._optimizers.items():
            optimizer_states[name] = optimizer.state_dict()

        state = {
            'state_dict': self._model.state_dict(),
            'optimizers': optimizer_states,
            'steps': self._steps
        }
        torch.save(state, path)
        print("Saved checkpoint to path:", path)

    def load_checkpoint(self, label=None, path=None):
        """
        Loads a checkpoint!
        By default, loads the one with the highest number of training iterations.
        """

        if path is None and label is None:
            path_choices = glob.glob(
                "{}/{}-*.ckpt".format(self._checkpoint_dir, self._experiment_name))
            if len(path_choices) == 0:
                print("No checkpoint found")
                return
            steps = []
            for choice in path_choices:
                prefix_len = len(
                    "{}/{}-".format(self._checkpoint_dir, self._experiment_name))
                suffix_len = len(".ckpt")
                string_steps = choice[prefix_len:-suffix_len]
                try:
                    steps.append(int(string_steps))
                except ValueError:
                    steps.append(-1)

            path = path_choices[np.argmax(steps)]
            expected_steps = np.max(steps)

            state = torch.load(path, map_location=self._device)
            assert state['steps'] == np.max(steps)
        elif path is None and label is not None:
            path = "{}/{}-{}.ckpt".format(self._checkpoint_dir,
                                          self._experiment_name, label)
            print(path)
            state = torch.load(path, map_location=self._device)
        elif path is not None:
            state = torch.load(path, map_location=self._device)
        else:
            assert False, "invalid arguments!"

        self._model.load_state_dict(state['state_dict'])

        for name, state_dict in state['optimizers'].items():
            self._optimizers[name].load_state_dict(state_dict)

        self._steps = state['steps']

        print("Loaded checkpoint from path:", path)
