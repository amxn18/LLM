from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback}\n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifierChain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = ' Write an appropriate response to this positive feedback. \n {feedback}',
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback. \n {feedback}',
    input_variables= ['feedback']
)
branchChain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find any appropriate sentiment")
)

chain = classifierChain | branchChain

result = chain.invoke({'feedback' : 'This is the onr of the best  product i have ever purchased, would love to use it again'})

print(result)