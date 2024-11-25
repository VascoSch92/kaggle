import logging
from functools import wraps

__all__ = ["LoggerSetup", "log_method_call"]


class LoggerSetup:
    """Implement logging for the subclasses."""

    def __init__(self) -> None:
        self._initialize_the_logger()

    def _initialize_the_logger(self):
        """Initialize the logger, create a console handler, and format the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s")

        ch.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(ch)


def log_method_call(func: "function") -> "function":
    """Log the name of a method when it's called."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assuming 'self' is the first argument
        self = args[0]
        output = func.__name__.replace("_", " ").lstrip().title()
        if hasattr(self, "logger"):
            # Log the method name using the class's logger
            self.logger.info(output)
        else:
            # Default behavior if no logger is set up
            print(output)
        return func(*args, **kwargs)

    return wrapper
