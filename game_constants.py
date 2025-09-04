from enum import Enum
# Game actions
class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4