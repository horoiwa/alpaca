from dataclasses import dataclass


@dataclass
class AdapterConfig:

    explainers: list = []

    objectives: list = []

    no_use_cols: list = []

    constraint_max_min: dict = {}

    constraint_categorical: dict = {}

    constraint_discrete: dict = {}

    constraint_sum_equal: dict = {}
