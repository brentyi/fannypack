import glob
import os

import numpy as np
import torch
import dill

from ._optimizer import _BuddyOptimizer


class _BuddyCheckpointing:
    """Private mixin for encapsulating checkpointing functions.
    """

    def __init__(self):
        """Checkpointing-specific setup.
        """
        super().__init__()

        # Find all unlabeled checkpoints for this experiment
        self._checkpointing_unlabeled_files = _BuddyCheckpointing._find_unlabeled_checkpoints(
            checkpoint_dir=self._config['checkpoint_dir'],
            experiment_name=self._experiment_name
        )

    def save_checkpoint(self, label=None, path=None):
        """Saves a checkpoint, which can optionally be labeled.
        """

        # Create directory if it doesn't exist yet
        if not os.path.isdir(self._config['checkpoint_dir']):
            os.makedirs(self._config['checkpoint_dir'])

        # Determine path to checkpoint file
        unlabeled = False
        if path is None and label is None:
            path = "{}/{}-{:016d}.ckpt".format(self._config['checkpoint_dir'],
                                               self._experiment_name, self._optimizer_steps)

            if path == self._checkpointing_unlabeled_files[-1]:
                self._print("Skipping redundant checkpoint save")
                return

            unlabeled = True
        elif path is None:
            # Numerical labels are reserved for step counts (see above)
            assert not label.isdigit()
            path = "{}/{}-{}.ckpt".format(self._config['checkpoint_dir'],
                                          self._experiment_name, label)

        # Create state to save. This includes:
        # > Model state
        # > Optimizers
        # > Training steps
        # > Buddy configuration
        optimizer_states = {}
        for name, optimizer in self._optimizer_dict.items():
            optimizer_states[name] = optimizer.state_dict()
        state = {
            'config': self._config,
            'state_dict': self._model.state_dict(),
            'optimizers': optimizer_states,
            'steps': self._optimizer_steps,
        }

        # "Atomic" checkpoint saving
        tmp_path = "/tmp/buddy-" + str(np.random.randint(1e10)) + ".ckpt"
        torch.save(state, tmp_path, pickle_module=dill)
        os.rename(tmp_path, path)
        self._print("Saved checkpoint to path:", path)

        # If unlabeled, add to list
        if unlabeled:
            self._checkpointing_unlabeled_files.append(path)

        # Prune checkpoint files
        while len(self._checkpointing_unlabeled_files) > \
                self._config['checkpoint_max_to_keep']:
            os.remove(self._checkpointing_unlabeled_files.pop(0))

    def load_checkpoint(self, label=None, path=None):
        """Loads a checkpoint. By default, loads the one with the highest
        number of training iterations.
        """

        # Determine path to checkpoint file
        if path is None and label is None:
            # Load latest unlabeled checkpoint
            if len(self._checkpointing_unlabeled_files) == 0:
                self._print("No checkpoint found")
                return False

            path = self._checkpointing_unlabeled_files[-1]
        elif path is None and label is not None:
            # Load a labeled checkpoint
            path = "{}/{}-{}.ckpt".format(self._config['checkpoint_dir'],
                                          self._experiment_name, label)
        elif path is not None:
            # Load a checkpoint by its location
            path = path
        else:
            assert False, "invalid arguments!"

        # Load checkpoint state
        state = torch.load(path, map_location=self._device, pickle_module=dill)

        # Sanity check: something's probably wrong if we're overwriting any
        # explicitly set, non-default configuration values
        for key, value in state['config'].items():
            assert state['config'][key] in (
                self._config[key], self.DEFAULT_CONFIG[key])

        # Load Buddy configuration
        self._config = state['config']

        # Load model parameters
        self._model.load_state_dict(state['state_dict'])

        # Load optimizer steps
        self._optimizer_steps = state['steps']

        # Instantiate optimizers and load their states
        self._optimizer_dict = _BuddyOptimizer._instantiate_optimizers(
            model=self._model,
            optimizer_type=self._config['optimizer_type'],
            optimizer_names=self._config['optimizer_names']
        )
        for name, state_dict in state['optimizers'].items():
            self._optimizer_dict[name].load_state_dict(state_dict)

        self._print("Loaded checkpoint from path:", path)
        return True

    @staticmethod
    def _find_unlabeled_checkpoints(checkpoint_dir, experiment_name):
        """(Private) Returns a list of all unlabeled checkpoints associated
        with this experiment, sorted from oldest to newest.
        """

        # Find all matching checkpoint files
        path_choices = glob.glob(
            "{}/{}-*.ckpt".format(checkpoint_dir, experiment_name))
        if len(path_choices) == 0:
            return []

        # Find unlabeled checkpoint files + associated step counts
        output = []
        for choice in path_choices:
            prefix_len = len("{}/{}-".format(checkpoint_dir, experiment_name))

            suffix_len = len(".ckpt")
            string_steps = choice[prefix_len:-suffix_len]
            try:
                steps = int(string_steps)
                output.append((choice, steps))
            except ValueError:
                pass

        # Sort output by steps
        output.sort(key=lambda x: x[1])

        # Return paths only
        return [x[0] for x in output]
