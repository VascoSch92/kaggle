from typing import Dict, Union, Optional

AGE_CATEGORIES = {
    "less than 20": {"less than 20", "15-20"},
    "20-24": {"20-25", "less than 20-25"},
    "25-29": {"25-30"},
    "30-34": {"30-35", "30-25"},
    "35-44": {"30-35", "30-40"},
    "45 and above": {"45 and above"},
}

NO_YES_CATEGORIES = {
    "yes": {"yes", "yes significantly", "yes, diagnosed by a doctor"},
    "no": {"no"},
    "missing": {"No, Yes, not diagnosed by a doctor"},
}

EXERCISE_TYPES_CATEGORIES = {
    "no exercise": {"no exercise"},
    "cardio": {"cardio (e.g., running, cycling, swimming)", "cardio (e.g.)"},
    "strength training": {
        "strength training (e.g., weightlifting, resistance exercises)",
        "strength training",
        "strength training (e.g.)",
    },
    "flexibility & balance": {"flexibility and balance (e.g., yoga, pilates)", "flexibility and balance (e.g.)"},
    "high-intensity training (hiit)": {"high-intensity interval training (hiit)"},
    "multiple exercise types": {
        "cardio, strength training",
        "cardio, flexibility and balance",
        "strength training, flexibility and balance",
        "cardio, strength training, flexibility and balance",
    },
    "missing": {None, "nan"},
}

SLEEP_CATEGORIES = {
    "insufficient sleep": {"less than 6 hours", "3-4 hours"},
    "normal sleep": {"6-8 hours"},
    "extended sleep": {"9-12 hours"},
    "excessive sleep": {"more than 12 hours"},
    "missing": {"nan"},
}


def categorize(
    element: Union[int, str, float],
    categories: Dict,
) -> Optional[Union[int, str, float]]:
    for key in categories.keys():
        if element in categories[key]:
            return key
    return None
