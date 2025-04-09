import random
from typing import List, Optional

from base_augmenter import BaseAxisAugmenter


class FewShotAugmenter(BaseAxisAugmenter):
    """
    Augmenter that handles few-shot examples for a given question.
    Controls the number and order of examples.
    """

    def __init__(self, min_examples: int = 1, max_examples: int = 5):
        """
        Initialize the few-shot augmenter.

        Args:
            min_examples: Minimum number of examples to use
            max_examples: Maximum number of examples to use
        """
        self.min_examples = min_examples
        self.max_examples = max_examples

    def augment(
        self, prompt: str, examples: List[str], shuffle: bool = False
    ) -> List[str]:
        """
        Generate variations of the prompt by adding few-shot examples.

        Args:
            prompt: The original question/prompt
            examples: List of example Q&A pairs
            shuffle: Whether to shuffle the examples or keep original order

        Returns:
            List of prompts with different few-shot configurations
        """
        if not examples:
            return [prompt]

        augmented_prompts = []

        for num_examples in range(
            self.min_examples, min(self.max_examples + 1, len(examples) + 1)
        ):
            if shuffle:
                # Create shuffled examples
                shuffled_examples = examples.copy()
                random.shuffle(shuffled_examples)
                selected_examples = shuffled_examples[:num_examples]
            else:
                # Original order
                selected_examples = examples[:num_examples]

            augmented_prompts.append(self._format_prompt(prompt, selected_examples))

        return augmented_prompts

    def _format_prompt(self, question: str, examples: List[str]) -> str:
        """
        Format the prompt with the few-shot examples.

        Args:
            question: The original question
            examples: List of examples to include

        Returns:
            Formatted prompt with few-shot examples
        """
        formatted_examples = "\n\n".join(examples)
        return f"{formatted_examples}\n\n{question}"


#
