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

    @property
    def covered_demos(self) -> List['DemoPair']:
        """Get all DemoPair instances that are covered by this demo's coverage graph."""
        covered_ids = self.meta.get('coverage', [])
        return [DemoPair.get_by_id(id) for id in covered_ids]
    
    @property
    def coverage_degree(self):
        return len(self.covered_demos())