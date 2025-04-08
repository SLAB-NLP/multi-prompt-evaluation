"""Constants for the Multi-Prompt Evaluation Tool."""

# Variation dimensions that can be selected
VARIATION_DIMENSIONS = [
    {
        "id": "enumeration",
        "name": "Enumeration Style",
        "description": "Change how options are labeled (A/B/C, 1/2/3, etc.)",
        "examples": ["A) Option", "1) Option", "• Option"]
    },
    {
        "id": "separator",
        "name": "Separators",
        "description": "Change how items are separated (commas, newlines, etc.)",
        "examples": ["Option, Option", "Option | Option", "Option\nOption"]
    },
    {
        "id": "order",
        "name": "Order",
        "description": "Change the order of options or information",
        "examples": ["A, B, C", "C, A, B", "B, C, A"]
    },
    {
        "id": "phrasing",
        "name": "Phrasing",
        "description": "Change how instructions are phrased",
        "examples": ["Select one option", "Choose one of the following", "Pick an option"]
    },
    {
        "id": "examples",
        "name": "Examples",
        "description": "Change the examples provided",
        "examples": ["Example X", "Example Y", "Example Z"]
    }
]

# Demo example for users to try
DEMO_EXAMPLE = """
# Customer Satisfaction Survey

Please rate your experience with our service:

A) Very satisfied
B) Satisfied
C) Neutral
D) Dissatisfied
E) Very dissatisfied

Please provide any additional comments below:
"""

# Demo dimensions to vary in the example
DEMO_DIMENSIONS = ["enumeration", "order", "phrasing"]

# Demo highlights in the example
DEMO_HIGHLIGHTS = [
    {
        "dimension": "enumeration",
        "start": 67,
        "end": 182,
        "text": "A) Very satisfied\nB) Satisfied\nC) Neutral\nD) Dissatisfied\nE) Very dissatisfied"
    },
    {
        "dimension": "phrasing",
        "start": 45,
        "end": 66,
        "text": "Please rate your experience with our service:"
    }
] 