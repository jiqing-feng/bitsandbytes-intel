print("Initializing bnb_intel module")


def _autoload():
    # need to import bitsandbytes to ensure the custom_op namespace is pre-populated
    import bitsandbytes  # noqa: F401

    from .ops import register_ops

    register_ops()
    return True


# Make _autoload available at the module level
__all__ = ["_autoload"]
