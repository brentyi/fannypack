import warnings


def deprecation_wrapper(message, function_or_class):
    """Creates a wrapper for a deprecated function or class. Prints a warning
    when the function or class is called.

    Args:
        message (str): Warning message.
        function_or_class (callable): Function or class to wrap.
    """

    def curried(*args, **kwargs):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return function_or_class(*args, **kwargs)

    return curried


def new_name_wrapper(old_name, new_name, function_or_class):
    """Creates a wrapper for a renamed function or class. Prints a warning when
    the function or class is called with the old name.

    Args:
        old_name (str): Old name of function or class. Printed in warning.
        new_name (str): New name of function or class. Printed in warning.
        function_or_class (callable): Function or class to wrap.
    """
    return deprecation_wrapper(
        f"{old_name} is deprecated! Use {new_name} instead.", function_or_class
    )
