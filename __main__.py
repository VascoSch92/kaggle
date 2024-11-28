import os

from tools.cli import validate_model, extract_arguments, execute_error_message
from tools.task import Pipeline
from titanic.etl import TitanicEtl
from titanic.train import TitanicTraining
from mental_health.etl import MentalHealthEtl
from mental_health.train import MentalHealthTrain


def main() -> None:
    project, model, others = extract_arguments()

    validate_model(model, project)
    os.environ["MODEL"] = model

    if others:
        for other in others:
            os.environ[other.replace("-", "").upper()] = other

    match project:
        case "titanic":
            match model:
                case "--preprocess":
                    Pipeline(TitanicEtl).run()
                case _, _:
                    Pipeline(TitanicEtl, TitanicTraining).run()
        case "mental-health":
            match model:
                case "--preprocess":
                    Pipeline(MentalHealthEtl).run()
                case _:
                    Pipeline(MentalHealthEtl, MentalHealthTrain).run()
        case _:
            execute_error_message(
                message=f"Project {project} not present.",
                exit_code=1,
            )


if __name__ == "__main__":
    main()
