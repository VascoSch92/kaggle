import sys
from typing import Set, List, Tuple, Optional
from pathlib import Path


def execute_error_message(message: str, exit_code: int) -> None:
    """Log an error message."""
    sys.stderr.write(f"ERROR: {message}.")
    sys.exit(exit_code)


def extract_arguments() -> Tuple[str, str, Optional[List[str]]]:
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
        case _:
            arguments = []
            for arg in sys.argv[2:]:
                if not arg.startswith("--"):
                    execute_error_message(
                        message=f"Expected flag of the form --FLAG, but got {arg}.",
                        exit_code=1,
                    )
                arguments.append(arg)

            if len(arguments) == 1:
                others = None
            else:
                others = arguments[1:]
            return sys.argv[1], sys.argv[2], others


def retrieve_valid_models(project: str) -> Set[str]:
    valid_pipelines = set()
    project_directory = Path().cwd() / project.replace("-", "_") / "models"
    if project_directory.exists() is False:
        return set()

    for p in project_directory.iterdir():
        candidate = p.stem
        if not candidate.startswith("__"):
            valid_pipelines.add("--" + candidate.replace("_", "-"))
    return valid_pipelines


def validate_model(model: str, project: str) -> None:
    valid_models = retrieve_valid_models(project=project)
    if model not in valid_models and valid_models:
        execute_error_message(
            message=f"Pipeline {model} not present.",
            exit_code=1,
        )
