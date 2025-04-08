from typing import List


class BaseAxisAugmenter:
    """
    Base class for all axis augmenters.
    
    Axis augmenters generate variations of a prompt along a specific dimension
    without changing the meaning of the prompt.
    """

    def __init__(self):
        """
        Initialize the augmenter with a name.
        
        Args:
            name: A descriptive name for this augmenter
        """
        pass

    def augment(self, prompt: str, identification_data: List[str]) -> List[str]:
        """
        Generate variations of the prompt based on identification data.
        
        Args:
            prompt: The original prompt text
            identification_data: Data from the identifier
            
        Returns:
            List of variations of this axis
        """
        # Default implementation returns the original prompt
        return [prompt]
