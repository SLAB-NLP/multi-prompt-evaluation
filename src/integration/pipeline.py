from typing import List, Dict, Any
import importlib
import os
import sys

from src.axis_identification.base_identifier import BaseAxisIdentifier
from src.axis_augmentation.base_augmenter import BaseAxisAugmenter

class AugmentationPipeline:
    """
    Simplified pipeline for prompt augmentation.
    
    Follows a sequential process:
    1. Axis Identification: Identify which aspects of an input can be varied
    2. Axis Augmentation: Generate variations for each identified aspect
    """
    
    def __init__(self):
        """Initialize the pipeline with the base identifier and augmenter."""
        self.identifier = BaseAxisIdentifier()
        self.augmenter = BaseAxisAugmenter()
    
    def load_components(self):
        """
        Load necessary components.
        This is kept for compatibility with existing code but doesn't do much now.
        """
        # Nothing to do in this simplified version
        pass
    
    def process(self, prompt: str) -> Dict[str, List[str]]:
        """
        Process a prompt through the pipeline.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            Dictionary mapping axis names to lists of variations
        """
        results = {}
        
        # Step 1: Axis Identification
        identification_data = self.identifier.identify(prompt)
        
        # If nothing was identified, return empty results
        if not identification_data:
            return results
            
        # Step 2: Axis Augmentation
        variations = self.augmenter.augment(prompt, identification_data)
        
        # Only include if there are actual variations
        if variations and len(variations) > 1:
            results[self.augmenter.get_name()] = variations
        
        return results