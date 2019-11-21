from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Dict, Any, Tuple


@dataclass_json
@dataclass
class AdapterConfig:
    """variable type maps
        categorical_label : label
        categorical_order : numerical
        numerical_float : numerical
        numrical_int : numerical
        no_use : too many nan
    """

    explainers: List[str] = None

    explainers_type: Dict[str, str] = None

    objectives: List[str] = None

    objectives_type: Dict[str, str] = None

    ml_explainers: List[str] = None

    ga_explainers: List[str] = None

    label_groups: Dict[str, List[str]] = None

    label_variables: Dict[str, str] = None

    class Meta:
        ordered = True


if __name__ == "__main__":
    import json
    import os
    import warnings
    warnings.filterwarnings('ignore')

    config = AdapterConfig()
    config.constraint_max_min = {"varA": [10, 100], "varB": [2, 100]}
    config.constraint_categorical = {"varC": ["panda", "deer", 33]}
    config.constraint_sum_equal = [{100: ["varD", "varE", "varF"]}]

    config_json = config.to_json(indent=4, ensure_ascii=False)
    config_json = json.loads(config_json)
    print(config_json)
    if os.path.exists("example/test.json"):
        os.remove("example/test.json")
    with open("example/test.json", "w") as f:
        json.dump(config_json, f)

    with open("example/test.json", "r") as f:
        json_config = f.read()

    config_2 = AdapterConfig.from_json(json_config)
    print(config_2)
    print(config_2.explainers is None)
