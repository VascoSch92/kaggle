import os
from typing import Set
from pathlib import Path

from tools.cli import extract_arguments, execute_error_message
from tools.task import Pipeline
from titanic.etl import TitanicEtl
from titanic.train import TitanicTraining
from mental_health.etl import MentalHealthEtl
from mental_health.train import MentalHealthTrain


def main() -> None:
    project, pipeline = extract_arguments()

    match project:
        case "titanic":
            valid_pipelines = retrieve_valid_pipeline(project=project)
            is_valid = pipeline in valid_pipelines

            match (pipeline, is_valid):
                case "--preprocess", _:
                    Pipeline(TitanicEtl).run()
                case _, False:
                    execute_error_message(
                        message=f"Pipeline {pipeline} not found.",
                        exit_code=1,
                    )
                case _, _:
                    os.environ["PIPELINE"] = pipeline
                    Pipeline(TitanicEtl, TitanicTraining).run()
        case "mental-health":
            valid_pipelines = retrieve_valid_pipeline(project=project)
            is_valid = pipeline in valid_pipelines

            match (pipeline, is_valid):
                case "--preprocess", _:
                    Pipeline(MentalHealthEtl).run()
                case _, False:
                    execute_error_message(
                        message=f"Pipeline {pipeline} not found.",
                        exit_code=1,
                    )
                case _, _:
                    os.environ["PIPELINE"] = pipeline
                    Pipeline(MentalHealthEtl, MentalHealthTrain).run()
        case _:
            execute_error_message(
                message=f"Project {project} not present.",
                exit_code=1,
            )


def retrieve_valid_pipeline(project: str) -> Set[str]:
    valid_pipelines = set()
    project_directory = Path().cwd() / project.replace("-", "_") / "models"
    for p in project_directory.iterdir():
        candidate = p.stem
        if not candidate.startswith("__"):
            valid_pipelines.add("--" + candidate.replace("_", "-"))
    return valid_pipelines


if __name__ == "__main__":
    main()
