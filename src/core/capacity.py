"""Native resource limit with byte counts for higher-level placement diagnostics."""


class CapacityExceeded(ValueError):
    def __init__(self, message, level, required, available):
        super().__init__(message)
        self.level = level
        self.required_bytes = required
        self.capacity_bytes = available
