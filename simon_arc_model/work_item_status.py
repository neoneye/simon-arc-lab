from enum import Enum

class WorkItemStatus(Enum):
    UNASSIGNED = 0
    UNVERIFIED = 1
    CORRECT = 2
    INCORRECT = 3
    PROBLEM_MISSING_PREDICTION_IMAGE = 4
    PROBLEM_DESERIALIZE = 5

    def to_string(self):
        if self == WorkItemStatus.PROBLEM_DESERIALIZE:
            return 'problemdeserialize'
        if self == WorkItemStatus.PROBLEM_MISSING_PREDICTION_IMAGE:
            return 'problemmissingpredictionimage'
        return self.name.lower()
