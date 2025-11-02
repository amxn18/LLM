from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Chat Template'

chatTemplate = ChatPromptTemplate([
    ('system', "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"), # All previous messages will be inserted here
    ('human', "{query}")
])

chat_history = []
# Load Chat History'

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())
print("Loaded Chat History:", chat_history)

# Create Prompt with Chat History'
prompt = chatTemplate.invoke({
    "chat_history": chat_history,
    "query": HumanMessage(content="Where is my Refund?")
})
print()
print(prompt)
