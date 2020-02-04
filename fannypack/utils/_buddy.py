import abc
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
import dill


class Buddy:
    """Buddy is a model manager that abstracts away PyTorch boilerplate.

    Helps with:
        - Creating/using/managing optimizers
        - Checkpointing (models + optimizers)
        - Namespaced/scoped Tensorboard logging
    """

    # Default configuration parameters
    DEFAULT_CONFIG = {
        'optimizer_type': "adam",
        'optimizer_names': ["primary"],
        'log_dir': "logs",
        'checkpoint_dir': "checkpoints",
        'checkpoint_max_to_keep': 5,
        'learning_rate_schedulers': {},
    }

    # Supported optimizer types
    OPTIMIZER_TYPES = {
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
    }
    DEFAULT_LEARNING_RATES = {
        'adam': 1e-4,
        'adadelta': 1
    }

    def __init__(self, experiment_name, model, load_checkpoint=True, **config):
        """Constructor
        """

        # Assign and validate core parameters
        assert type(experiment_name) == str
        assert isinstance(model, nn.Module)
        self._experiment_name = experiment_name
        self._model = model

        # Assign and validate our configuration
        self._config = self.DEFAULT_CONFIG.copy()
        for key, value in config.items():
            assert key in self.DEFAULT_CONFIG.keys()
            assert type(value) == type(self.DEFAULT_CONFIG[key])
            self._config[key] = config[key]

        # Create some misc state variables for tensorboard
        # The writer is lazily instantiated in TrainingBuddy.log()
        self._writer = None
        self._log_scopes = []

        # What device are we using for training?
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        print("Using device:", self._device)
        torch.autograd.set_detect_anomaly(True)

        # Instantiate optimizers, step count -- note that these may be
        # overriden by our loaded checkpoint
        self._optimizers = self._instantiate_optimizers()
        self._steps = 0

        # Load checkpoint using model name
        self._unlabeled_checkpoint_files = self._find_unlabeled_checkpoints()
        if load_checkpoint:
            self.load_checkpoint()

    def set_learning_rate(self, value, optimizer_name="primary"):
        """Sets an optimizer learning rate. Accepts either a floating point
        learning rate or a schedule function (int steps -> float LR).
        """

        schedulers = self._config['learning_rate_schedulers']
        if callable(value):
            # Store a scheduler
            assert type(value(0)) == float
            schedulers[optimizer_name] = value
        else:
            # Delete scheduler
            if optimizer_name in schedulers:
                schedulers.pop(optimizer_name)

            # Set scalar learning rate
            self._set_learning_rate(value, optimizer_name)

    def _set_learning_rate(self, value, optimizer_name):
        """(Private) Sets an optimizer's learning rate.
        """

        assert optimizer_name in self._optimizers.keys()

        optimizer = self._optimizers[optimizer_name]
        for param_group in optimizer.param_groups:
            param_group['lr'] = value

    def minimize(self, loss, optimizer_name="primary",
                 retain_graph=False, checkpoint_interval=1000):
        """Compute gradients and use them to minimize a loss function.
        """

        assert optimizer_name in self._optimizers.keys()

        # Update learning rate using scheduler if possible
        schedulers = self._config['learning_rate_schedulers']
        if optimizer_name in schedulers:
            self._set_learning_rate(
                schedulers[optimizer_name](self._steps), optimizer_name)

        # Take gradient step
        self._optimizers[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizers[optimizer_name].step()

        # Update step & checkpoint
        self._steps += 1
        if self._steps % checkpoint_interval == 0:
            self.save_checkpoint()

    def log_scope(self, scope):
        """Returns a scope to log tensors in.

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
        """Log a tensor for visualization in Tensorboard. Currently only supports scalars.
        """
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(
                self._config['log_dir'] + "/" + self._experiment_name)

        self._writer.add_scalar(name, value, global_step=self._steps)

    def save_checkpoint(self, label=None, path=None):
        """Saves a checkpoint!
        """

        # Create directory if it doesn't exist yet
        if not os.path.isdir(self._config['checkpoint_dir']):
            os.makedirs(self._config['checkpoint_dir'])

        unlabeled = False
        if path is None and label is None:
            path = "{}/{}-{:016d}.ckpt".format(self._config['checkpoint_dir'],
                                               self._experiment_name, self._steps)
            unlabeled = True
        elif path is None:
            # Numerical labels are reserved for step counts (see above)
            assert not label.isdigit()
            path = "{}/{}-{}.ckpt".format(self._config['checkpoint_dir'],
                                          self._experiment_name, label)

        # Create state to save
        optimizer_states = {}
        for name, optimizer in self._optimizers.items():
            optimizer_states[name] = optimizer.state_dict()
        state = {
            'state_dict': self._model.state_dict(),
            'optimizers': optimizer_states,
            'steps': self._steps,
            'config': self._config
        }

        # "Atomic" saving
        tmp_path = "/tmp/buddy-" + str(np.random.randint(1e10)) + ".ckpt"
        torch.save(state, tmp_path, pickle_module=dill)
        os.rename(tmp_path, path)
        print("Saved checkpoint to path:", path)

        # If unlabeled, add to list
        if unlabeled:
            self._unlabeled_checkpoint_files.append(path)

        # Prune checkpoint files
        while len(self._unlabeled_checkpoint_files) > \
                self._config['checkpoint_max_to_keep']:
            os.remove(self._unlabeled_checkpoint_files.pop(0))

    def load_checkpoint(self, label=None, path=None):
        """Loads a checkpoint!
        By default, loads the one with the highest number of training iterations.
        """

        if path is None and label is None:
            # Load latest unlabeled checkpoint
            print(self._unlabeled_checkpoint_files)
            if len(self._unlabeled_checkpoint_files) == 0:
                print("No checkpoint found")
                return False
            path = self._unlabeled_checkpoint_files[-1]
        elif path is None and label is not None:
            # Load a labeled checkpoint
            path = "{}/{}-{}.ckpt".format(self._config['checkpoint_dir'],
                                          self._experiment_name, label)
        elif path is not None:
            # Load a checkpoint by its location
            path = path
        else:
            assert False, "invalid arguments!"

        state = torch.load(path, map_location=self._device, pickle_module=dill)

        # Sanity check: something's probably wrong if we're overwriting any
        # explicitly set, non-default configuration values
        for key, value in state['config'].items():
            assert state['config'][key] in (
                self._config[key], self.DEFAULT_CONFIG[key])

        # Load model parameters
        self._model.load_state_dict(state['state_dict'])
        self._steps = state['steps']
        self._config = state['config']

        # Instantiate optimizers and set their states
        self._optimizers = self._instantiate_optimizers()
        for name, state_dict in state['optimizers'].items():
            self._optimizers[name].load_state_dict(state_dict)

        print("Loaded checkpoint from path:", path)
        return True

    def _find_unlabeled_checkpoints(self):
        """Returns a list of all unlabeled checkpoints associated with this experiment,
        sorted from oldest to newest.
        """

        # Find all matching checkpoint files
        path_choices = glob.glob(
            "{}/{}-*.ckpt".format(self._config['checkpoint_dir'], self._experiment_name))
        if len(path_choices) == 0:
            return []

        # Find unlabeled checkpoint files + associated step counts
        output = []
        for choice in path_choices:
            prefix_len = len(
                "{}/{}-".format(self._config['checkpoint_dir'], self._experiment_name))
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

    def _instantiate_optimizers(self):
        """Private method for instantiating optimizer objects, setting default learning rates.
        """

        # Make sure we're creating a valid optimizer
        optimizer_type = self._config['optimizer_type']
        assert optimizer_type in self.OPTIMIZER_TYPES

        # Instantiate an optimizer for each value in _config['optimizer_names']
        #
        # Note that if we're loading from a checkpoint, the initial learning
        # rate may be immediately overwritten
        Optimizer = self.OPTIMIZER_TYPES[optimizer_type]
        initial_learning_rate = self.DEFAULT_LEARNING_RATES[optimizer_type]
        optimizer_instances = {}
        for name in self._config['optimizer_names']:
            optimizer_instances[name] = Optimizer(
                self._model.parameters(), lr=initial_learning_rate)

        return optimizer_instances
