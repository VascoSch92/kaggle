import sys
from typing import Tuple


def execute_error_message(message: str, exit_code: int) -> None:
    """Log an error message."""
    sys.stderr.write(f"ERROR: {message}.")
    sys.exit(exit_code)


def extract_arguments() -> Tuple[str, str]:
    """Extract arguments from command line."""
    arguments = sys.argv
    number_of_arguments = len(arguments) - 1

    match number_of_arguments:
        case 0:
            execute_error_message(
                message="Expected 2 arguments, but 0 where given.",
                exit_code=1,
            )
        case 1:
            execute_error_message(
                message="Expected 2 arguments, but 1 where given.",
                exit_code=1,
            )
        case 2:
            if not sys.argv[2].startswith("--"):
                execute_error_message(
                    message=f"Expected flag of the form --PIPELINE_NAME, but got {sys.argv[2]}.",
                    exit_code=1,
                )
            return sys.argv[1], sys.argv[2]
        case _:
            execute_error_message(
                message=f"Expected 2 arguments, but {number_of_arguments} were given.",
                exit_code=1,
            )
