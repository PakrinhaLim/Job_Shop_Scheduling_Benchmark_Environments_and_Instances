from typing import List
from scheduling_environment.operation import Operation


class Job:
    def __init__(self, job_id: int):
        self._job_id: int = job_id
        self._operations: List[Operation] = []
        self._due_date: float = 0.0

    def __repr__(self):
        return (
            f"<Job(job_id={self._job_id})>"
        )

    def add_operation(self, operation: Operation):
        """Add an operation to the job."""
        self._operations.append(operation)

    @property
    def nr_of_ops(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def operations(self) -> List[Operation]:
        """Return the list of operations."""
        return self._operations

    @property
    def job_id(self) -> int:
        """Return the job's id."""
        return self._job_id

    @property
    def scheduled_operations(self) -> List[Operation]:
        """Return a list of operations that have been scheduled to a machine."""
        return [operation for operation in self._operations if operation.scheduling_information != {}]

    @property
    def due_date(self) -> float:
        """Return the job's due date."""
        return self._due_date

    @due_date.setter
    def due_date(self, value: float):
        """Set the job's due date."""
        self._due_date = value

    @property
    def tardiness(self) -> float:
        """Return the tardiness of the job."""
        if not self._operations or self._operations[-1].scheduled_end_time is None:
            return 0.0
        return max(0.0, self._operations[-1].scheduled_end_time - self._due_date)