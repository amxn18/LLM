# LangChain Prompts: 

1️⃣ LangChain Static Prompts

# Static prompts are the simplest form of interaction with a language model.

Definition: A static prompt is a fixed instruction or query that does not change dynamically.

Use-case: When you want to consistently query the model in the same format, for example summarizing a research paper with a fixed template.

Key characteristics:
- Input is directly embedded in the prompt.
- Does not rely on user selections or variables.
- Easy to implement, ideal for simple applications.

Workflow:
1. Receive user input (e.g., a research paper name).
2. Send the fixed prompt to the model with the user input interpolated.
3. Retrieve the response and display it.

Benefit: Quick setup for basic summarization or repetitive tasks.

2️⃣ LangChain Dynamic Prompts

# Dynamic prompts allow for flexible and customizable interactions.

Definition: A dynamic prompt incorporates variables and templates that can be filled based on user input or programmatic conditions.

Use-case: When you need the model’s response to adapt to user-selected options, such as explanation style, length, or other criteria.

Key characteristics:
- Uses PromptTemplate or load_prompt() to define placeholders for variables.
- User input dynamically populates placeholders (e.g., paper title, explanation style, length).
- Enables richer, multi-dimensional querying compared to static prompts.

Workflow:
1. Display selectable options for users.
2. Populate a prompt template dynamically with user selections.
3. Send the constructed prompt to the language model.
4. Render the model’s output in a readable format.

Benefit: Highly flexible, supports multiple user-driven configurations.

3️⃣ Single List of Messages

# LangChain can manage conversations as a list of message objects, allowing context-aware interactions.

Definition: Each interaction is treated as a structured message, categorized as system, human, or AI messages.

Message types:
- SystemMessage: Defines rules, behavior, or persona of the AI.
- HumanMessage: Represents user input.
- AIMessage: Captures model output.

Use-case: Real-time chat systems where prior messages must be remembered for context.

Workflow:
1. Initialize a chat history with system messages.
2. Append each user input as a HumanMessage.
3. Send the full chat history to the language model.
4. Append model responses as AIMessage objects.
5. Continue appending to maintain full conversational context.

Benefit: Ensures the model generates responses that are contextually relevant to the conversation history.

4️⃣ LangChain Messages

# Messages in LangChain provide structured input and output handling for chat-based interactions.

Purpose: To define clear roles and track message flow between user and model.

Key points:
- Messages allow the model to differentiate between instructions (system), queries (human), and responses (AI).
- You can send multiple messages together, allowing more complex prompts with layered instructions.
- Messages can be appended or combined to build a coherent interaction flow.

Benefit: Provides structure and clarity in multi-turn conversations, reducing ambiguity in model responses.

5️⃣ LangChain Message Holders

# Message holders allow you to store and reuse previous conversation history in prompts.

Definition: A mechanism to hold multiple messages and inject them dynamically into a chat template.

Key components:
- ChatPromptTemplate: A high-level template that can include message placeholders.
- MessagesPlaceholder: Marks the location where past conversation history is inserted.
- chat_history: A list of past messages or interactions to be included in the current prompt.

Workflow:
1. Define a chat template with system instructions and placeholders for previous messages.
2. Load or maintain chat_history from previous interactions.
3. Fill the template dynamically with chat_history and current user query.
4. Send the assembled prompt to the language model for response.

Benefit: Enables context-preserving prompts and supports dynamic multi-turn interactions.

🔹 Summary of Concepts

| Concept                 | Purpose                     | Key Characteristics                     | Use-Case                          |
|-------------------------|----------------------------|-----------------------------------------|----------------------------------|
| Static Prompt           | Fixed, simple instruction  | Hardcoded input                          | Quick summarization              |
| Dynamic Prompt          | Flexible, template-based   | Uses variables, user selections          | Adaptive explanations            |
| Single List of Messages | Context-aware chat         | System, Human, AI messages in a list    | Real-time conversational AI      |
| Messages                | Structured input/output    | Differentiates roles                      | Multi-turn or layered instructions|
| Message Holders         | Store/reuse chat history   | MessagesPlaceholder + ChatPromptTemplate | Context-preserving prompts       |
