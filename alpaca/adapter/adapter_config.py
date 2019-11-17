from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AdapterConfig:

    explainers: List[str] = field(default_factory=list)

    objectives: List[str] = field(default_factory=list)

    no_use_cols: List[str] = field(default_factory=list)

    constraint_max_min: Dict[str, List[int]] = field(default_factory=dict)

    constraint_discrete: Dict[str, List[int]] = field(default_factory=dict)

    constraint_categorical: Dict[str, List[Any]] = field(default_factory=dict)

    constraint_sum_equal: List[Dict[int, List[str]]] = field(default_factory=list)

    class Meta:
        ordered = True


if __name__ == "__main__":
    import marshmallow_dataclass
    import json
    import os

    AdapterSchema = marshmallow_dataclass.class_schema(AdapterConfig)
    config = AdapterConfig()
    config.constraint_max_min["varA"] = [10, 100]
    config.constraint_categorical["varB"] = ["panda", 100]
    config.constraint_sum_equal["varC"] = [{100, ["varD", "varE", "varF"]}]

    config_json = AdapterSchema().dump(config)

    print(config_json)
    print(type(config_json))
    if os.path.exists("example/test.json"):
        os.remove("example/test.json")

    with open("example/test.json", "w") as f:
        json.dump(config_json, f, indent=4)

    with open("example/test.json", "r") as f:
        json_config = json.load(f)

    config_2 = AdapterSchema().load(json_config)
    print(config_2)
