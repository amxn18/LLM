from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",    
    temperature=0.4
)

model = ChatHuggingFace(llm=llm)

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

# Schema --> JSON 
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structuredModel = model.with_structured_output(json_schema)
#result = structuredModel.invoke(review1)
result = structuredModel.invoke(review2)
print(result)
print(type(result))

# If you’re looking for full built-in structured-output support (i.e., strictly schema-adhering responses via with_structured_output() or similar), you will get the most reliable behavior with commercial APIs (OpenAI, Anthropic, etc) or endpoints that explicitly support function calling / JSON schema.

#If you stick with open-source models, you can get structured-like output, but you’ll often need to use wrappers or simulation techniques (e.g., JSONFormer, manual prompt + parsing) rather than rely on built-in support.

# For your learning/project work: you should plan for the fact that open-source models may require post-processing/parsing, retries or fallback logic rather than guarantee schema compliance.