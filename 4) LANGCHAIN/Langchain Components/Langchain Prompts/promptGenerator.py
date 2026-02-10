from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template=(
        "Please summarize the research paper titled '{paper_input}' with the following specifications:\n"
        "- Explanation Style: {style_input}\n"
        "- Explanation Length: {length_input}\n\n"
        "Include relevant mathematical or code explanations when applicable.\n"
        "If the information is missing, respond with 'Insufficient information available.'"
    ),
    input_variables=['paper_input', 'style_input', 'length_input']
)
validate_template = True

template.save('template.json')