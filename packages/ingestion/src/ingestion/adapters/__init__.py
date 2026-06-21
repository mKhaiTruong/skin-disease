from abc import ABC, abstractmethod

class BaseIngestionAdapter(ABC):
    @abstractmethod
    def fetch(self, dst: str) -> None:
        pass