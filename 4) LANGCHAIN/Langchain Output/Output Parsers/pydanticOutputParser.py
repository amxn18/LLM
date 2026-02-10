from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task = "text-generation",
    temperature = 0.3
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person) 

template = PromptTemplate(
    template = 'Generate the name, age and city of a {origin} sports athlete \n {format_instructions}',
    input_variables=["origin"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'origin' : 'Indian'})
print(result)