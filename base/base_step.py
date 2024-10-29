class BaseStep:
    def execute(self) -> None:
        raise NotImplementedError("Subclasses must implement execute method")