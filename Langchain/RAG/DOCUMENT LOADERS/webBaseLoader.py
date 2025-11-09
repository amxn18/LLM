from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = 'https://www.flipkart.com/apple-macbook-air-m3-8-gb-256-gb-ssd-macos-sonoma-mrxv3hn-a/p/itm5c4be2e024c52'
loader = WebBaseLoader(url)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question:\n{question}\n\nUsing this text:\n{text}',
    input_variables=['question', 'text']
)

loadChain = RunnableLambda(lambda _: loader.load()[0].page_content)

questionChain = RunnableLambda(lambda _: "What is the price of this product ?")

prepareInputs = RunnableParallel({
    "text": loadChain,
    "question": questionChain
})

chain = prepareInputs | prompt | model | parser

result = chain.invoke(None)
print(result)

# Output --> The price of the Apple MacBook Air M3 with 16GB memory, 512GB SSD, and macOS Sonoma is ₹1,34,900. 