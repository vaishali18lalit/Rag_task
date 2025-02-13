def build_prompt(parsed_results, user_question):
    """Build a prompt for the multimodal LLM with both text and image content for each year."""

    # Initialize the prompt content with user question
    prompt_content = f"Answer the question based only on the following context from 2023 and 2024 reports:\n\n"

    # Add 2023 context (texts and images)
    prompt_content += "Context from the 2023 report:\n"
    if parsed_results["2023"]["texts"]:
        prompt_content += "\n".join([text for text in parsed_results["2023"]["texts"]]) + "\n"
    else:
        prompt_content += "No text content available for 2023.\n"
    
    if parsed_results["2023"]["images"]:
        prompt_content += "\nImages from the 2023 report:\n"
        for img in parsed_results["2023"]["images"]:
            prompt_content += f"[Image: {img}]\n"
    else:
        prompt_content += "No image content available for 2023.\n"

    # Add 2024 context (texts and images)
    prompt_content += "\nContext from the 2024 report:\n"
    if parsed_results["2024"]["texts"]:
        prompt_content += "\n".join([text for text in parsed_results["2024"]["texts"]]) + "\n"
    else:
        prompt_content += "No text content available for 2024.\n"

    if parsed_results["2024"]["images"]:
        prompt_content += "\nImages from the 2024 report:\n"
        for img in parsed_results["2024"]["images"]:
            prompt_content += f"[Image: {img}]\n"
    else:
        prompt_content += "No image content available for 2024.\n"

    # Append the user's question
    prompt_content += f"\nQuestion: {user_question}\n"

    return prompt_content
