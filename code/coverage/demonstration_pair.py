from typing import List, Dict, Any
class DemoPair:
    """Representation of a single demonstration pair."""
    all_instances: List['DemoPair'] = []

    def __init__(self, input: str, output: str, id: int, meta: Dict[str, Any] = None):
        self.input = input
        self.output = output
        self.id = id
        self.meta = meta or {}
        DemoPair.all_instances.append(self)

    @classmethod
    def get_by_id(cls, id: int) -> 'DemoPair':
        for instance in cls.all_instances:
            if instance.id == id:
                return instance
        raise ValueError(f"No DemoPair found with id {id}")
    
    @staticmethod
    def to_dict(demo: 'DemoPair') -> dict:
        """Convert DemoPair to dictionary."""
        return {
            "input": demo.input,
            "output": demo.output,
            "id": demo.id,
            "meta": demo.meta
        }

    @staticmethod
    def from_dict(data: dict) -> 'DemoPair':
        """Create DemoPair from dictionary."""
        return DemoPair(
            input=data["input"],
            output=data["output"],
            id=data["id"],
            meta=data["meta"]
        )
