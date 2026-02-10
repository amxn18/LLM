# LANGCHAIN CHAINS 

────────────────────────────────────────────────────────
# 1. SIMPLE CHAIN
────────────────────────────────────────────────────────

A Simple Chain is the most basic building block in LangChain.  
It connects components in a linear order using the pipe operator `|`.

Structure:
Prompt → Model → Parser

Use Case:
- When you need a single input, single output pipeline.
- When output from the model directly completes the task.

Key Characteristics:
- Executes exactly one prompt.
- No intermediate logic or branching.
- Best suited for single-purpose tasks (fact generation, Q&A, rewriting, etc.).

Example Flow:
1. Provide a topic to the prompt.
2. Model generates text.
3. StrOutputParser converts output to a clean string.

Advantages:
- Minimal setup.
- Low latency.
- Highly predictable execution.

Limitations:
- Cannot handle multi-step workflows.
- No memory of previous outputs unless manually passed.

────────────────────────────────────────────────────────
# 2. SEQUENTIAL CHAIN
────────────────────────────────────────────────────────

A Sequential Chain passes the output of one step into the next step in order.

Structure:
Step 1 → Step 2 → Step 3 → ... → Final Output

Use Case:
- Multi-step text generation.
- When each stage depends on the previous stage.
- Examples:
  - Report → Summary → Title
  - Document → Notes → Quiz → Final Output

Key Characteristics:
- Fully linear workflow.
- Each stage depends on previous results.
- Built by chaining multiple (Prompt → Model → Parser) units.

Example Flow:
1. Prompt1 creates a detailed report.
2. Prompt2 turns the report into bullet points.
3. Prompt3 creates a title from bullet points.

Advantages:
- Human-like multi-step processing.
- Highly modular (easy to add/remove stages).
- Best for document transformation workflows.

Limitations:
- Slower because stages run one after another.
- Errors in one step propagate forward.

────────────────────────────────────────────────────────
# 3. PARALLEL CHAIN
────────────────────────────────────────────────────────

A Parallel Chain executes multiple tasks simultaneously using `RunnableParallel`.

Structure:
           → Task A →
Input ────             ───→ Merge Step → Final Output
           → Task B →

Use Case:
- When two independent tasks must run at the same time.
- Example:
  - Notes generation + Quiz generation from the same text.
  - Summary + Keywords extraction.
  - Classification + Title generation (when independent).

Key Characteristics:
- Executes tasks concurrently.
- Returns outputs as a dictionary (e.g., { notes: ..., quiz: ... }).
- Must be followed by a merge step if combined output is required.

Example Flow:
1. parallel_chain generates:
   - notes
   - quiz
2. merge_chain combines notes and quiz into final study material.

Advantages:
- Much faster than sequential chaining.
- Efficient when tasks do not depend on each other.

Limitations:
- Parallel tasks must be independent.
- Cannot parallelize dependent tasks (e.g., quiz_from_notes cannot run parallel).

────────────────────────────────────────────────────────
# 4. CONDITIONAL CHAIN
────────────────────────────────────────────────────────

A Conditional Chain chooses one of several possible execution paths based on runtime conditions.

Implemented using:
- RunnableBranch
- RunnableLambda

Structure:
Input → Classifier → Condition → Chosen Branch → Output

Use Case:
- Dynamic workflows.
- Sentiment-based routing.
- Safety checks.
- Multi-action agents.

Example Flow:
1. classifierChain determines sentiment: positive or negative.
2. RunnableBranch selects:
   - For positive: positive-response prompt.
   - For negative: negative-response prompt.
   - Default: fallback response.

Advantages:
- Enables decision-making workflows.
- High flexibility and real-world applicability.
- Allows classification + action sequences.

Limitations:
- Classifier must return correct structured output.
- Branch definitions must match exact return types.

────────────────────────────────────────────────────────
# RUNNABLES SYSTEM (ESSENTIAL OVERVIEW)
────────────────────────────────────────────────────────

LangChain’s new architecture is based on **Runnables**, which represent anything executable:

Types used here:
1. RunnableParallel  
2. RunnableBranch  
3. RunnableLambda  
4. RunnableSerializable (base for all composed chains)

Purpose:
- To unify prompts, models, parsers, functions, and logic under one universal interface.
- Every component can be chained with `|` regardless of type.

Important Behaviors:
- All runnable objects support:
   - invoke()
   - batch()
   - stream()
   - graph visualization (get_graph())

RunnableParallel:
- Executes tasks in parallel.
- Returns a dictionary of results.
- Used for multi-output pipelines.

RunnableBranch:
- Implements if-else logic.
- Routes input to different chains based on conditions.

RunnableLambda:
- Wraps Python functions inside chain logic.
- Used for fallback logic or dynamic decision making.

Advantages of Runnables:
- Unified API for all components.
- Better composition than old LCEL chains.
- Full support for parallel, sequential, and branching logic.
- Works with all models and parsers.

────────────────────────────────────────────────────────
# COMPARISON TABLE
────────────────────────────────────────────────────────

| Chain Type        | Execution Pattern       | Use Case                                 | Supports Parallelism | Supports Branching |
|-------------------|--------------------------|-------------------------------------------|------------------------|---------------------|
| Simple Chain      | Single linear step       | Direct single-step generation             | No                     | No                  |
| Sequential Chain  | Multi-step linear        | Reports, summaries, multi-stage tasks     | No                     | No                  |
| Parallel Chain    | Concurrent execution     | Independent tasks (notes + quiz)          | Yes                    | No                  |
| Conditional Chain | Dynamic branching        | Sentiment routing, decision workflows     | No                     | Yes                 |

────────────────────────────────────────────────────────
# SUMMARY
────────────────────────────────────────────────────────

- Use **Simple Chain** for one-shot LLM tasks.
- Use **Sequential Chain** when later steps depend on earlier outputs.
- Use **Parallel Chain** when tasks can run independently.
- Use **Conditional Chain** for decision-based or classification-based routing.
- Runnables provide a unified, composable system that supports linear, parallel, and branching workflows.
