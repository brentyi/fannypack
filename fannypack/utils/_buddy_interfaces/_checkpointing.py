import glob
import os
import pathlib
import signal
import warnings

import numpy as np
import torch
import dill

from ._optimizer import _BuddyOptimizer


class _BuddyCheckpointing:
    """Buddy's model checkpointing interface.
    """

    def __init__(self, checkpoint_dir, checkpoint_max_to_keep):
        """Checkpointing-specific setup.
        """
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_max_to_keep = checkpoint_max_to_keep

        # Find all unlabeled checkpoints for this experiment
        self._checkpoint_unlabeled_files = self._find_unlabeled_checkpoints(
            checkpoint_dir=self._checkpoint_dir,
            experiment_name=self._experiment_name,
        )

    def save_checkpoint(self, label=None, path=None):
        """Saves a checkpoint, which can optionally be labeled.
        """

        # Determine path to checkpoint file
        unlabeled = False
        if path is None and label is None:
            path = "{}/{}-{:016d}.ckpt".format(
                self._checkpoint_dir,
                self._experiment_name,
                self._optimizer_steps,
            )

            if (
                self._checkpoint_unlabeled_files
                and path == self._checkpoint_unlabeled_files[-1]
            ):
                self._print("Skipping redundant checkpoint save")
                return

            unlabeled = True
        elif path is None:
            # Numerical labels are reserved for step counts (see above)
            assert not label.isdigit()
            path = "{}/{}-{}.ckpt".format(
                self._checkpoint_dir, self._experiment_name, label
            )

        # Create directory if it doesn't exist yet
        checkpoint_dir = pathlib.Path(path).parents[0]
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Create state to save. This includes:
        # > Model state
        # > Optimizers
        # > Training steps
        # > Buddy configuration
        optimizer_states = {}
        for name, optimizer in self._optimizer_dict.items():
            optimizer_states[name] = optimizer.state_dict()
        state = {
            "config": self._optimizer_config,
            "state_dict": self._model.state_dict(),
            "optimizers": optimizer_states,
            "steps": self._optimizer_steps,
        }

        # Ignore SIGINT (eg ctrl+c) events while we save to disk...
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError as e:
            # signal throws a ValueError if we're not in the main thread
            self._print("Error while attaching SIGINT handler:", e)
            orig_handler = None

        # "Atomic" checkpoint saving
        tmp_path = "{}/tmp-{}.ckpt".format(
            checkpoint_dir, np.random.randint(1e10),
        )
        torch.save(state, tmp_path, pickle_module=dill)
        os.rename(tmp_path, path)
        self._print("Saved checkpoint to path:", path)

        # Restore SIGINT handler
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)

        # If unlabeled, add to list
        if unlabeled:
            self._checkpoint_unlabeled_files.append(path)

        # Prune checkpoint files
        while (
            len(self._checkpoint_unlabeled_files)
            > self._checkpoint_max_to_keep
        ):
            os.remove(self._checkpoint_unlabeled_files.pop(0))

    def load_checkpoint_module(
        self, source, target=None, label=None, path=None, experiment_name=None
    ):
        """Loads parameters from a specific child module within a checkpoint.
        By default, loads the checkpoint with the highest number of training
        iterations.

        Can also be specified via a label or file path.
        """

        if target is None:
            target = source

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)
        if checkpoint is None:
            return

        # Get possible target modules
        module_dict = dict(self._model.named_modules())
        assert target in module_dict.keys(), "Nonexistent target module!"

        # Build a state dict for this module only
        source_state_dict = {}
        key_prefix = ""
        if len(source) > 0:
            key_prefix = f"{source}."
        for key, value in checkpoint["state_dict"].items():
            if key.startswith(key_prefix):
                source_state_dict[key[len(key_prefix) :]] = value

        # Load state dict
        missing, unexpected = module_dict[target].load_state_dict(
            source_state_dict
        )
        assert len(missing) == 0
        assert len(unexpected) == 0

        self._print(f"Loaded module: {source} => {target}")

    def load_checkpoint_optimizer(
        self, source, target=None, label=None, path=None, experiment_name=None
    ):
        """Loads state associated with a specific optimizer from a checkpoint.
        By default, loads the checkpoint with the highest number of training
        iterations.

        Can also be specified via a label or file path.
        """

        if target is None:
            target = source

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)
        if checkpoint is None:
            return

        # Sanity check
        assert (
            source in checkpoint["optimizers"].keys()
        ), "Nonexistent source optimizer!"
        assert (
            target in self._optimizer_dict.keys()
        ), "Nonexistent target optimizer!"

        # Load optimizer state
        state_dict = checkpoint["optimizers"][source]
        self._optimizer_dict[target].load_state_dict(state_dict)
        self._print(f"Loaded optimizer: {source} => {target}")

    def load_checkpoint_optimizers(
        self, label=None, path=None, experiment_name=None
    ):
        """Loads all optimizer settings from a checkpoint. By default, loads
        the checkpoint with the highest number of training iterations.

        Can also be specified via a label or file path.
        """

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)
        if checkpoint is None:
            return

        # Load optimizer state
        self._load_checkpoint_optimizers(checkpoint)

    def load_checkpoint(self, label=None, path=None, experiment_name=None):
        """Loads a checkpoint. By default, loads the one with the highest
        number of training iterations.

        Can also be specified via a label or file path.
        """

        # Find and read our checkpoint file
        checkpoint = self._read_checkpoint_file(label, path, experiment_name)
        if checkpoint is None:
            return

        # Load optimizer state
        self._load_checkpoint_optimizers(checkpoint)

        # Load model parameters
        missing, unexpected = self._model.load_state_dict(
            checkpoint["state_dict"]
        )
        assert len(missing) == 0
        assert len(unexpected) == 0

        # Load optimizer steps
        self._optimizer_steps = checkpoint["steps"]

        self._print("Loaded checkpoint at step:", self._optimizer_steps)
        return True

    @property
    def checkpoint_labels(self):
        """ Accessor for listing available checkpoint labels.
        """

        experiment_name = self._experiment_name
        checkpoint_dir = self._checkpoint_dir

        # Find all matching checkpoint files
        path_choices = glob.glob(
            "{}/{}-*.ckpt".format(checkpoint_dir, experiment_name)
        )
        if len(path_choices) == 0:
            return []

        # Pull out labels
        output = []
        for choice in path_choices:
            prefix_len = len("{}/{}-".format(checkpoint_dir, experiment_name))

            suffix_len = len(".ckpt")
            string_label = choice[prefix_len:-suffix_len]
            output.append(string_label)

        # Sort output alphabetically and return
        output.sort()
        return output

    def _load_checkpoint_optimizers(self, checkpoint):
        # Load Buddy optimizer configuration
        for key, value in checkpoint["config"].items():
            if key not in self._optimizer_config.keys():
                warnings.warn(
                    f"Skipping invalid configuration key: {key}={value}"
                )
                continue
            self._optimizer_config[key] = value

        # Instantiate optimizers
        self._optimizer_dict = _BuddyOptimizer._instantiate_optimizers(
            model=self._model,
            optimizer_type=self._optimizer_config["optimizer_type"],
            optimizer_names=self._optimizer_config["optimizer_names"],
        )

        # Load optimizer states
        for name, state_dict in checkpoint["optimizers"].items():
            self._optimizer_dict[name].load_state_dict(state_dict)

    def _read_checkpoint_file(self, label, path, experiment_name):
        """Find a checkpoint to load.

        This is one of three options:
          1) The latest, based on step # (must have same experiment name)
          2) A file saved with a label (must have same experiment name)
          3) One specified by a path
        """

        # Determine path to checkpoint file
        if path is None and label is None:
            # Load latest unlabeled checkpoint
            if experiment_name is None:
                # Use our current experiment name by default
                paths = self._checkpoint_unlabeled_files
            else:
                # Use specified experiment name
                paths = self._find_unlabeled_checkpoints(
                    checkpoint_dir=self._checkpoint_dir,
                    experiment_name=experiment_name,
                )
            if len(paths) == 0:
                raise FileNotFoundError("Failed to find checkpoint file")

            # The list of paths will be sorted by optimizer step count
            path = paths[-1]

        elif path is None and label is not None:
            # Load a labeled checkpoint
            if experiment_name is None:
                # Use our current experiment name by default
                experiment_name = self._experiment_name
            path = "{}/{}-{}.ckpt".format(
                self._checkpoint_dir, experiment_name, label
            )
        elif path is not None:
            # Load a checkpoint by its location
            path = path
        else:
            assert False, "invalid arguments!"

        # Load and return checkpoint dict
        checkpoint = torch.load(
            path, map_location=self._device, pickle_module=dill
        )

        # Sanity check: our checkpoint file is a sensible-looking dict
        assert set(checkpoint.keys()) == set(
            ["config", "state_dict", "optimizers", "steps"]
        )

        # Sanity check: something's probably wrong if we're overwriting any
        # explicitly set, non-default configuration values
        # for key, value in checkpoint['config'].items():
        #     assert checkpoint['config'][key] in (
        #         self._optimizer_config[key], self.DEFAULT_CONFIG[key])

        # Sanity check: optimizer names and type should typically be consistent
        if checkpoint["optimizers"].keys() != self._optimizer_dict.keys():
            warnings.warn("Checkpoint loading: overriding optimizer names.")
            warnings.warn("Before: " + str(self._optimizer_dict.keys()))
            warnings.warn("After: " + str(checkpoint["optimizers"].keys()))
        if (
            checkpoint["config"]["optimizer_type"]
            != self._optimizer_config["optimizer_type"]
        ):
            warnings.warn("Checkpoint loading: overriding optimizer type.")

        self._print("Read checkpoint from path:", path)
        return checkpoint

    @staticmethod
    def _find_unlabeled_checkpoints(checkpoint_dir, experiment_name):
        """(Private) Returns a list of all unlabeled checkpoints associated
        with this experiment, sorted from oldest to newest.
        """

        # Find all matching checkpoint files
        path_choices = glob.glob(
            "{}/{}-*.ckpt".format(checkpoint_dir, experiment_name)
        )
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
