import pandas as pd
import streamlit as st

from src.integration.combinatorial import VariationCombiner
from src.integration.pipeline import AugmentationPipeline


def main():
    """Streamlit app for Multi-Prompt Evaluation Tool."""
    st.title("Multi-Prompt Evaluation Tool")
    st.write("""
    This tool generates variations of prompts without changing their meaning,
    allowing for more robust evaluation of language models.
    """)

    # Input section
    st.header("Input")
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "CSV Upload"]
    )

    prompt = ""
    if input_method == "Text Input":
        prompt = st.text_area("Enter your prompt:", height=200)
    else:
        st.write("Please upload a CSV file with a 'prompt' column containing the prompts to process.")
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'prompt' in df.columns:
                    prompt = df['prompt'].iloc[0]  # Take the first prompt
                    st.text_area("First prompt from CSV:", prompt, height=200)
                    st.info(f"CSV contains {len(df)} rows. Only processing the first prompt.")
                else:
                    st.error("CSV file must contain a 'prompt' column.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    # Processing parameters
    st.header("Processing Parameters")
    max_combinations = st.slider("Maximum number of combinations:", 1, 500, 100)

    # Process button
    process_button = st.button("Process Prompt")

    if process_button and prompt:
        with st.spinner("Processing..."):
            # Initialize the pipeline
            pipeline = AugmentationPipeline()
            pipeline.load_components()

            # Process the prompt
            variations_by_axis = pipeline.process(prompt)

            # Display results
            st.header("Results")

            if not variations_by_axis:
                st.warning("No variation axes were identified in the prompt.")
            else:
                st.success(f"Found {len(variations_by_axis)} axes that can be varied.")

                # Display variations by axis
                st.subheader("Variations by Axis")
                for axis_name, variations in variations_by_axis.items():
                    with st.expander(f"{axis_name} ({len(variations)} variations)"):
                        for i, var in enumerate(variations):
                            st.text_area(f"Variation {i + 1}", var, height=100)

                # Generate combinations
                combiner = VariationCombiner(max_combinations=max_combinations)
                combined_variations = combiner.combine(variations_by_axis)

                # Display combined variations
                st.subheader(f"Combined Variations ({len(combined_variations)} total)")

                # Create a dataframe for easier viewing
                df = pd.DataFrame({
                    "Variation #": range(1, len(combined_variations) + 1),
                    "Text": combined_variations
                })

                st.dataframe(df)

                # Download button for results
                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='prompt_variations.csv',
                    mime='text/csv',
                )

    # Show original prompt for reference
    if prompt:
        st.sidebar.header("Original Prompt")
        st.sidebar.text_area("", prompt, height=300)


if __name__ == "__main__":
    main()
