from typing import List

from dataset.demonstration_pair import DemoPair


class Function:
	def __init__(self):
		self.demo_pairs: List[DemoPair] = []
		self._seen_inputs = set()

	def _ensure_unique_input(self, demo_pair: DemoPair) -> None:
		if demo_pair.input in self._seen_inputs:
			raise ValueError(f"Duplicate input is not allowed: {demo_pair.input}")

	def add_demo_pair(self, demo_pair: DemoPair) -> None:
		self._ensure_unique_input(demo_pair)
		self.demo_pairs.append(demo_pair)
		self._seen_inputs.add(demo_pair.input)

	def extend(self, demo_pairs: List[DemoPair]) -> None:
		for demo_pair in demo_pairs:
			self.add_demo_pair(demo_pair)

	def get_by_input(self, input_text: str) -> None | DemoPair:
		for demo_pair in self.demo_pairs:
			if demo_pair.input == input_text:
				return demo_pair
		return None

	def __len__(self) -> int:
		return len(self.demo_pairs)
