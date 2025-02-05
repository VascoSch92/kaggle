import os

from tools.cli import validate_model, extract_arguments, execute_error_message
from tools.task import Pipeline
from insurance.etl import InsuranceEtl
from insurance.train import InsuranceTrain
from mental_health.etl import MentalHealthEtl
from health_factors.etl import HealthFactorsEtl
from product_defect.etl import ProductDefectEtl
from mental_health.train import MentalHealthTrain
from health_factors.train import HealthFactorsTrain
from product_defect.train import ProductDefectTrain
from spaceship_titanic.etl import SpaceshipTitanicEtl
from spaceship_titanic.train import SpaceshipTitanicTrain


def main() -> None:
    project, model, others = extract_arguments()

    validate_model(model, project)
    os.environ["MODEL"] = model

    if others:
        for other in others:
            os.environ[other.replace("-", "").upper()] = other

    match project:
        case "mental-health":
            match model:
                case "--etl":
                    Pipeline(MentalHealthEtl).run()
                case _:
                    Pipeline(MentalHealthEtl, MentalHealthTrain).run()
        case "spaceship-titanic":
            match model:
                case "--etl":
                    Pipeline(SpaceshipTitanicEtl).run()
                case _:
                    Pipeline(SpaceshipTitanicEtl, SpaceshipTitanicTrain).run()
        case "insurance":
            match model:
                case "--etl":
                    Pipeline(InsuranceEtl).run()
                case _:
                    Pipeline(InsuranceEtl, InsuranceTrain).run()
        case "product-defect":
            match model:
                case "--etl":
                    Pipeline(ProductDefectEtl).run()
                case _:
                    Pipeline(ProductDefectEtl, ProductDefectTrain).run()
        case "health-factors":
            match model:
                case "--etl":
                    Pipeline(HealthFactorsEtl).run()
                case _:
                    Pipeline(HealthFactorsEtl, HealthFactorsTrain).run()
        case _:
            execute_error_message(
                message=f"Project {project} not present.",
                exit_code=1,
            )


if __name__ == "__main__":
    main()
