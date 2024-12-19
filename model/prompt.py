FAVA_PROMPT_TEMPLATE = (
    "Read the following references:\n"
    "{evidence}\n"
    "Please identify all the errors in the following text using the information in the references provided and suggest edits if necessary:\n"
    "[Text] {output}\n"
    "[Edited] "
)

REFIND_PROMPT_TEMPLATE_WO_QUESTION = (
    "Refer to the references below and generate information that you can get from the references.\n\n"
    "### References\n"
    "{references}\n"
    "### Information\n"
)

REFIND_PROMPT_TEMPLATE = (
    "You are an assistant for answering questions.\n"
    "Refer to the references below and answer the following question.\n\n"
    "### References\n"
    "{references}\n\n"
    "### Question\n"
    "{question}\n"
    "### Answer\n"
)

