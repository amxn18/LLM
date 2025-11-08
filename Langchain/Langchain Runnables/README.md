# LANGCHAIN RUNNABLES

────────────────────────────────────────────────────────
# 1. WHAT ARE RUNNABLES?
────────────────────────────────────────────────────────

Runnables are the core execution units of LangChain’s modern system.  
A Runnable is any object that:
- Accepts an input
- Performs an operation (LLM call, parsing, function call, routing, parallel execution)
- Returns an output
- Can be chained using the `|` operator

Runnables unify all components (prompts, models, parsers, Python functions, branching, parallelism) under a single interface.

Every runnable supports:
- invoke()
- batch()
- stream()
- composition with `|`
- graph visualization using get_graph()

This makes Runnables a universal abstraction for all pipelines.

────────────────────────────────────────────────────────
# 2. TWO TYPES OF RUNNABLES
────────────────────────────────────────────────────────

Runnables are divided into:

1. Type-Specific Runnables  
   These are components that represent a specific operation:
   - PromptTemplate
   - LLMs (ChatHuggingFace, ChatOpenAI, etc.)
   - OutputParsers
   - Custom functions (RunnableLambda)
   - Branch logic (RunnableBranch)
   - Passthrough connectors (RunnablePassthrough)

2. Runnable Primitives  
   These describe how to run a sequence or structure:
   - RunnableSequence
   - RunnableParallel
   - RunnableBranch (logic primitive)
   - RunnableLambda
   - RunnablePassthrough
   - RunnableMap (mapping over dict fields)
   - RunnableEach (mapping over a list)

Type-specific runnables define WHAT to do.  
Runnable primitives define HOW to run them (sequence, parallel, branching, etc.)

────────────────────────────────────────────────────────
# 3. LCEL AND RUNNABLESEQUENCE
────────────────────────────────────────────────────────

Before LCEL, LangChain workflows used `RunnableSequence()` to explicitly chain steps.

Example:
RunnableSequence(prompt, model, parser)

After LCEL (LangChain Expression Language):
- You can write chains using the `|` operator:
  prompt | model | parser
- LCEL automatically builds a RunnableSequence internally.
- Developers no longer need to manually write RunnableSequence.

This drastically simplifies code.

Old:
seq = RunnableSequence(step1, step2, step3)
New:
chain = step1 | step2 | step3

RunnableSequence still exists for learning purposes, but LCEL is now the standard.

────────────────────────────────────────────────────────
# 4. RUNNABLE PRIMITIVES 
────────────────────────────────────────────────────────

Below are all runnable primitives you used, with detailed understanding and exact behavior.

────────────────────────────────────────────────────────
## 4.1 RUNNABLESEQUENCE — Linear Execution

Executes each step in order.  
Input → Step 1 → Step 2 → Step 3 → Output

Used when each stage depends on output from the previous stage.

CODE EXAMPLE:
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
model = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen3-4B-Instruct-2507",task="text-generation"))
parser = StrOutputParser()
prompt2 = PromptTemplate(template='Explain the following joke - {text}', input_variables=['text'])

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
result = chain.invoke({'topic': "AI"})

EXPLANATION:
- Step 1: prompt creates a joke.
- Step 2: model generates output.
- Step 3: parser parses it.
- Step 4: prompt2 asks explanation.
- Step 5: model explains.
- Step 6: parser outputs text.

────────────────────────────────────────────────────────
## 4.2 RUNNABLEPARALLEL — Parallel Execution

Runs multiple chains simultaneously.  
Input → [Task A || Task B] → Output as dictionary {a: ..., b: ...}

CODE EXAMPLE:
parallelChain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model1, parser),
    'linkedin': RunnableSequence(prompt2, model2, parser)
})
result = parallelChain.invoke({'topic': "LSTM in Neural Networks"})

EXPLANATION:
- Both tweet and linkedin post are generated at the same time.
- Final output is a dictionary:
  { "tweet": "...", "linkedin": "..." }

Use when tasks do NOT depend on each other.

────────────────────────────────────────────────────────
## 4.3 RUNNABLEPASSTHROUGH — Pass Input Without Modification

Returns the input exactly as received.  
Useful in parallel chains or merging steps.

CODE EXAMPLE:
parallelChain = RunnableParallel({
    'Joke': RunnablePassthrough(),
    'Explanation': RunnableSequence(prompt2, model, parser)
})

EXPLANATION:
Joke returns original input.
Explanation generates explanation, producing combined output:
{ "Joke": <joke>, "Explanation": <explanation> }

────────────────────────────────────────────────────────
## 4.4 RUNNABLELAMBDA — Pure Python Logic Inside Chain

Wraps a Python function to compute values or transformations.

CODE EXAMPLE:
def countWords(text): return len(text.split())

parallelChain = RunnableParallel({
    'Joke': RunnablePassthrough(),
    'Words': RunnableLambda(countWords)
})

result becomes:
{ "Joke": "...", "Words": 12 }

EXPLANATION:
RunnableLambda allows mixing LLM logic and Python logic.

────────────────────────────────────────────────────────
## 4.5 RUNNABLEBRANCH — Conditional Chain Logic

Works like IF / ELSE for runnable pipelines.

Branch structure:
RunnableBranch(
    (condition, chain_if_true),
    (condition, chain_if_false),
    default_chain
)

CODE EXAMPLE:
branchChain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(generateReport, branchChain)

EXPLANATION:
- If report > 500 words → summarize it.
- Else → return report unchanged.

This enables intelligent routing inside pipelines.

────────────────────────────────────────────────────────
# 5. COMPARATIVE TABLE OF RUNNABLE PRIMITIVES
────────────────────────────────────────────────────────

| Primitive            | Purpose                       | Input Dependency | Most Useful For |
|----------------------|-------------------------------|------------------|------------------|
| RunnableSequence     | Linear step-by-step chain     | Yes              | Multi-step processing |
| RunnableParallel     | Run tasks simultaneously      | No               | Independent tasks |
| RunnablePassthrough  | Return input unchanged        | N/A              | Merging, parallel work |
| RunnableLambda       | Add Python logic              | N/A              | Computation, metrics |
| RunnableBranch       | Conditional routing           | Yes              | Decision workflows |

────────────────────────────────────────────────────────
# 6. LCEL vs RUNNABLESEQUENCE
────────────────────────────────────────────────────────

Before LCEL:
chain = RunnableSequence(prompt, model, parser)

After LCEL:
chain = prompt | model | parser

LCEL:
- More readable
- Automatically constructs a RunnableSequence behind the scenes
- Works with all runnable types
- Enables mixing logic, branches, parallelism with minimal syntax

LCEL is now the recommended way to build pipelines.

────────────────────────────────────────────────────────
# 7. SUMMARY
────────────────────────────────────────────────────────

- Runnables unify the execution model of LangChain.
- Type-specific runnables define operations.
- Runnable primitives define execution strategy (sequence, parallel, branching).
- LCEL replaces explicit RunnableSequence creation.
- Runnables allow building any AI workflow: linear, parallel, conditional, hybrid.

