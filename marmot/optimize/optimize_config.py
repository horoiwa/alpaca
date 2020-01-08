from dataclasses import dataclass
from dataclass_json import dataclass_json
from typing import List, Dict


@dataclass_json
@dataclass
class OptimizationConfig:

    all_variables = List[str]

    group_variables: Dict[str, str]

    groups: List[List[str]]

    group_constraints: Dict[str, str]

    independent_variables: Dict[str, str]

    independent_constraints: Dict[str, str]
