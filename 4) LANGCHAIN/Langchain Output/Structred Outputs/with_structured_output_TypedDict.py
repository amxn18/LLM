from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    temperature=0.4
)
review1 = """The hadware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also the UI looks outdated compared other brands. Hoping for a software update to fix this."""

review2 = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""
chat = ChatHuggingFace(llm=llm)
# Schema --> TypedDict
class Review(TypedDict):
    keyThemes : Annotated[list[str], "Write down all the key themes discused in the review in a list."]
    summary: Annotated[str, "A concise summary of the review in one sentence."]
    sentiment: Annotated[str, "The overall sentiment of the review, either Positive, Negative, or Mixed."]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


structuredModel = chat.with_structured_output(Review)
#result = structuredModel.invoke(review1)
result = structuredModel.invoke(review2)
print(result)

print(type(result))

# Output: --> Review 1
# {'summary': "The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also the UI looks outdated compared to other brands.", 'sentiment': 'Mixed'}


# Output: --> Review 2
#{'keyThemes': ['Samsung Galaxy S24 Ultra', 'Snapdragon 8 Gen 3 processor', 'gaming', 'multitasking', 'editing photos', '5000mAh battery', '45W fast charging', 'S-Pen integration', '200MP camera', 'night mode', '100x zoom', 'One UI', 'bloatware', '$1,300 price tag', 'one-handed use'], 
# 'summary': 'The Samsung Galaxy S24 Ultra is an impressive device with a powerful processor, excellent camera, and long battery life, but is large, heavy, and expensive with unnecessary bloatware. ', 
# 'sentiment': 'Mixed', 
# 'pros': ['Insanely powerful processor (great for gaming and productivity)', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful'], 
# 'cons': ['Large and heavy, uncomfortable for one-handed use', 'Bloatware with Samsung apps', '$1,300 price tag']}
