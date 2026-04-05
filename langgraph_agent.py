import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    generation: str

    # Router result: apple / tesla / both / none
    route_target: str

    # Original single-path fields (keep these for non-comparison questions)
    documents: str
    search_count: int
    needs_rewrite: str

    # Comparison branch: Apple
    apple_query: str
    apple_documents: str
    apple_grade: str
    apple_search_count: int

    # Comparison branch: Tesla
    tesla_query: str
    tesla_documents: str
    tesla_grade: str
    tesla_search_count: int


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    # Router prompt for Task B:
    # classify the question into apple / tesla / both / none
    options = list(FILES.keys()) + ["both", "none"]
    router_prompt = f"""
You are a financial question router.

Your task is to classify the user question into exactly one of these four categories:
{', '.join(options)}

Routing rules:
1. Output ONLY valid JSON.
2. The JSON format must be exactly: {{"datasource": "..."}}
3. Valid datasource values are only: "apple", "tesla", "both", or "none".
4. Choose "apple" if the question is only about Apple.
5. Choose "tesla" if the question is only about Tesla.
6. Choose "both" if the question compares Apple and Tesla, or clearly asks about both companies.
7. Choose "none" if the question is unrelated to Apple or Tesla financial documents.

Examples:
Question: What was Apple's revenue in 2024?
Output: {{"datasource": "apple"}}

Question: What was Tesla's net income in 2024?
Output: {{"datasource": "tesla"}}

Question: Compare Apple and Tesla revenue in 2024.
Output: {{"datasource": "both"}}

Question: What is NVIDIA's revenue in 2024?
Output: {{"datasource": "none"}}

User Question: {question}
"""

    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()

        # Handle markdown code fences if the model wraps JSON output
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        res_json = json.loads(content)
        target = res_json.get("datasource", "none")
    except Exception as e:
        print(colored(f"⚠️ Error parsing router output: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"

    if target not in options:
        print(colored(f"⚠️ Invalid router label '{target}'. Defaulting to 'none'.", "yellow"))
        target = "none"

    print(colored(f"🎯 Routing to: {target}", "cyan"))

    docs_content = ""

    # Keep retrieval behavior simple and avoid doing rewrite-like
    # financial term mapping inside the retrieve node.
    company_queries = {}

    if target == "apple":
        company_queries["apple"] = question

    elif target == "tesla":
        company_queries["tesla"] = question

    elif target == "both":
        # Do not rewrite or expand the comparison query here.
        # The compare-specific queries will be prepared later in the compare branch.
        pass

    # target == "none" -> no retrieval
    for t, query_to_use in company_queries.items():
        if t in RETRIEVERS:
            print(colored(f"   Query for {t}: {query_to_use}", "light_cyan"))

            docs = RETRIEVERS[t].invoke(query_to_use)
            source_name = t.capitalize()

            docs_content += f"\n\n[Source: {source_name}]\n"
            docs_content += "\n".join([d.page_content for d in docs])

    return {
        "route_target": target,
        "documents": docs_content,
        "search_count": state["search_count"] + 1
    }

@retry_logic
def prepare_compare_queries(state: AgentState):
    print(colored("--- 🧩 PREPARING COMPARE QUERIES ---", "blue"))

    question = state["question"]

    # Extract year if present
    year = None
    for y in ["2024", "2023", "2022"]:
        if y in question:
            year = y
            break

    question_lower = question.lower()

    # Build simple company-specific queries without hard-mapping
    # Keep the original metric wording as much as possible.
    if "apple" in question_lower and "tesla" in question_lower:
        # Remove the explicit compare phrasing and split into two company-specific questions
        base_question = question

        # Make the Apple-side query
        apple_query = base_question
        apple_query = apple_query.replace("Apple and Tesla", "Apple")
        apple_query = apple_query.replace("Tesla and Apple", "Apple")
        apple_query = apple_query.replace("apple and tesla", "Apple")
        apple_query = apple_query.replace("tesla and apple", "Apple")
        apple_query = apple_query.replace("Compare ", "")
        apple_query = apple_query.replace("compare ", "")
        apple_query = apple_query.replace("versus", "")
        apple_query = apple_query.replace("vs.", "")
        apple_query = apple_query.replace("vs", "")
        apple_query = apple_query.strip()

        # Make the Tesla-side query
        tesla_query = base_question
        tesla_query = tesla_query.replace("Apple and Tesla", "Tesla")
        tesla_query = tesla_query.replace("Tesla and Apple", "Tesla")
        tesla_query = tesla_query.replace("apple and tesla", "Tesla")
        tesla_query = tesla_query.replace("tesla and apple", "Tesla")
        tesla_query = tesla_query.replace("Compare ", "")
        tesla_query = tesla_query.replace("compare ", "")
        tesla_query = tesla_query.replace("versus", "")
        tesla_query = tesla_query.replace("vs.", "")
        tesla_query = tesla_query.replace("vs", "")
        tesla_query = tesla_query.strip()

    else:
        # Fallback:
        # If the wording is unusual, just prepend the company names to the original question.
        apple_query = f"Apple {question}"
        tesla_query = f"Tesla {question}"

    print(colored(f"   Apple query: {apple_query}", "light_cyan"))
    print(colored(f"   Tesla query: {tesla_query}", "light_cyan"))

    return {
        "apple_query": apple_query,
        "tesla_query": tesla_query,
        "apple_search_count": 0,
        "tesla_search_count": 0,
        "apple_grade": "no",
        "tesla_grade": "no",
        "apple_documents": "",
        "tesla_documents": "",
    }

@retry_logic
def retrieve_apple_compare(state: AgentState):
    print(colored("--- 🍎 RETRIEVING APPLE FOR COMPARISON ---", "blue"))

    apple_query = state["apple_query"]
    docs_content = ""

    if "apple" not in RETRIEVERS:
        print(colored("⚠️ Apple retriever not found.", "yellow"))
        return {
            "apple_documents": "",
            "apple_search_count": state["apple_search_count"] + 1
        }

    print(colored(f"   Apple query: {apple_query}", "light_cyan"))

    docs = RETRIEVERS["apple"].invoke(apple_query)

    docs_content += "\n\n[Source: Apple]\n"
    docs_content += "\n".join([d.page_content for d in docs])

    return {
        "apple_documents": docs_content,
        "apple_search_count": state["apple_search_count"] + 1
    }

@retry_logic
def grade_apple_compare(state: AgentState):
    print(colored("--- 🍎⚖️ GRADING APPLE FOR COMPARISON ---", "yellow"))

    question = state["question"]
    apple_query = state["apple_query"]
    apple_documents = state["apple_documents"]
    llm = get_llm()

    # Hard guard:
    # If Apple retrieval returned nothing, Apple side is not sufficient.
    if not apple_documents.strip():
        print("   Apple Grade: no (empty context)")
        return {"apple_grade": "no"}

    system_prompt = """You are a strict financial relevance grader for the Apple side of a comparison task.

Your task is to decide whether the retrieved Apple document context is sufficient for answering the Apple side of the user's question.

Rules:
1. Output ONLY one word: yes
2. Or output ONLY one word: no
3. Do not output any explanation.
4. Say yes only if the Apple context contains useful and sufficient evidence.
5. Say no if the Apple context is empty, unrelated, too vague, or does not support answering the Apple side of the question.
"""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Original comparison question:\n{question}\n\n"
                f"Apple-side retrieval query:\n{apple_query}\n\n"
                f"Retrieved Apple context:\n{apple_documents}"
            )
        )
    ]

    response = llm.invoke(msg)
    content = response.content.strip().lower()

    # Safer parsing: only accept exact yes / no
    if content == "yes":
        grade = "yes"
    elif content == "no":
        grade = "no"
    else:
        grade = "no"

    print(f"   Apple Grade: {grade}")
    return {"apple_grade": grade}

@retry_logic
def rewrite_apple_compare(state: AgentState):
    print(colored("--- 🍎🔄 REWRITING APPLE QUERY FOR COMPARISON ---", "red"))

    question = state["question"]
    apple_query = state["apple_query"]
    llm = get_llm()

    msg = [
        SystemMessage(
            content="""You are a financial query rewriter for the Apple side of a comparison task.

Your task is to rewrite the Apple-side query so it becomes easier for a financial document retriever to answer.

Rules:
1. Keep the original meaning.
2. Focus only on Apple.
3. Make the Apple query more specific and retrieval-friendly.
4. Prefer precise financial terminology when appropriate, such as:
   - revenue
   - net sales
   - net income
   - operating income
   - research and development expenses
   - capital expenditures
   - total assets
   - cash flow from operating activities
5. Preserve years such as 2024, 2023, or 2022 if they are mentioned.
6. Do not invent new years or new facts.
7. Output ONLY the rewritten Apple query text, with no explanation.

Examples:
Original query: company profit in 2024
Rewritten query: company net income in 2024

Original query: company new tech spending in 2024
Rewritten query: company research and development expenses in 2024

Original query: company operating cash flow in 2024
Rewritten query: company cash flow from operating activities in 2024
"""
        ),
        HumanMessage(
            content=(
                f"Original comparison question:\n{question}\n\n"
                f"Current Apple-side query:\n{apple_query}"
            )
        )
    ]

    response = llm.invoke(msg)
    new_apple_query = response.content.strip()

    print(f"   New Apple query: {new_apple_query}")

    return {"apple_query": new_apple_query}

@retry_logic
def retrieve_tesla_compare(state: AgentState):
    print(colored("--- 🚗 RETRIEVING TESLA FOR COMPARISON ---", "blue"))

    tesla_query = state["tesla_query"]
    docs_content = ""

    if "tesla" not in RETRIEVERS:
        print(colored("⚠️ Tesla retriever not found.", "yellow"))
        return {
            "tesla_documents": "",
            "tesla_search_count": state["tesla_search_count"] + 1
        }

    print(colored(f"   Tesla query: {tesla_query}", "light_cyan"))

    docs = RETRIEVERS["tesla"].invoke(tesla_query)

    docs_content += "\n\n[Source: Tesla]\n"
    docs_content += "\n".join([d.page_content for d in docs])

    return {
        "tesla_documents": docs_content,
        "tesla_search_count": state["tesla_search_count"] + 1
    }

@retry_logic
def grade_tesla_compare(state: AgentState):
    print(colored("--- 🚗⚖️ GRADING TESLA FOR COMPARISON ---", "yellow"))

    question = state["question"]
    tesla_query = state["tesla_query"]
    tesla_documents = state["tesla_documents"]
    llm = get_llm()

    # Hard guard:
    # If Tesla retrieval returned nothing, Tesla side is not sufficient.
    if not tesla_documents.strip():
        print("   Tesla Grade: no (empty context)")
        return {"tesla_grade": "no"}

    system_prompt = """You are a strict financial relevance grader for the Tesla side of a comparison task.

Your task is to decide whether the retrieved Tesla document context is sufficient for answering the Tesla side of the user's question.

Rules:
1. Output ONLY one word: yes
2. Or output ONLY one word: no
3. Do not output any explanation.
4. Say yes only if the Tesla context contains useful and sufficient evidence.
5. Say no if the Tesla context is empty, unrelated, too vague, or does not support answering the Tesla side of the question.
"""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Original comparison question:\n{question}\n\n"
                f"Tesla-side retrieval query:\n{tesla_query}\n\n"
                f"Retrieved Tesla context:\n{tesla_documents}"
            )
        )
    ]

    response = llm.invoke(msg)
    content = response.content.strip().lower()

    # Safer parsing: only accept exact yes / no
    if content == "yes":
        grade = "yes"
    elif content == "no":
        grade = "no"
    else:
        grade = "no"

    print(f"   Tesla Grade: {grade}")
    return {"tesla_grade": grade}

@retry_logic
def rewrite_tesla_compare(state: AgentState):
    print(colored("--- 🚗🔄 REWRITING TESLA QUERY FOR COMPARISON ---", "red"))

    question = state["question"]
    tesla_query = state["tesla_query"]
    llm = get_llm()

    msg = [
        SystemMessage(
            content="""You are a financial query rewriter for the Tesla side of a comparison task.

Your task is to rewrite the Tesla-side query so it becomes easier for a financial document retriever to answer.

Rules:
1. Keep the original meaning.
2. Focus only on Tesla.
3. Make the Tesla query more specific and retrieval-friendly.
4. Prefer precise financial terminology when appropriate, such as:
   - revenue
   - net income
   - operating income
   - research and development expenses
   - capital expenditures
   - total assets
   - cash flow from operating activities
5. Preserve years such as 2024, 2023, or 2022 if they are mentioned.
6. Do not invent new years or new facts.
7. Output ONLY the rewritten Tesla query text, with no explanation.

Examples:
Original query: company profit in 2024
Rewritten query: company net income in 2024

Original query: company new tech spending in 2024
Rewritten query: company research and development expenses in 2024

Original query: company operating cash flow in 2024
Rewritten query: company cash flow from operating activities in 2024
"""
        ),
        HumanMessage(
            content=(
                f"Original comparison question:\n{question}\n\n"
                f"Current Tesla-side query:\n{tesla_query}"
            )
        )
    ]

    response = llm.invoke(msg)
    new_tesla_query = response.content.strip()

    print(f"   New Tesla query: {new_tesla_query}")

    return {"tesla_query": new_tesla_query}

@retry_logic
def compare_generate(state: AgentState):
    print(colored("--- ⚖️✍️ GENERATING COMPARISON ANSWER ---", "green"))

    question = state["question"]
    apple_documents = state["apple_documents"]
    tesla_documents = state["tesla_documents"]
    llm = get_llm()

    # Hard guard:
    # If either side is missing, do not compare from model memory.
    if not apple_documents.strip() or not tesla_documents.strip():
        return {"generation": "I don't know."}

    combined_context = f"{apple_documents}\n\n{tesla_documents}"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a financial analyst.

Answer the user's comparison question using ONLY the provided context from Apple and Tesla.

Strict rules:
1. The final answer must be in English only.
2. Use ONLY the provided context.
3. Do not use outside knowledge.
4. Do not guess or add unsupported interpretation.
5. If both sides of the comparison are not clearly supported by the context, say exactly: I don't know.
6. Keep the answer concise and direct.
7. Always cite the source in brackets, such as [Source: Apple] and [Source: Tesla].
8. For a comparison question, clearly mention both companies in the answer.

Preferred answer style:
- One or two short sentences.
- State the value for Apple and the value for Tesla if available.
- Then briefly state the comparison result.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": combined_context,
        "question": question
    })

    answer = response.content.strip()

    # Safety check:
    # If the model gives a non-empty answer without any source citation
    # and it is not exactly "I don't know.", force a safer fallback.
    if answer != "I don't know." and "[Source:" not in answer:
        answer = "I don't know."

    return {"generation": answer}

def compare_fail(state: AgentState):
    print(colored("--- ❌ COMPARISON FAILED ---", "red"))
    return {"generation": "I don't know."}

@retry_logic
def grade_documents_node(state: AgentState):
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # Hard guard 1:
    # If there is no retrieved context at all, it cannot be relevant.
    if not documents.strip():
        print("   Relevance Grade: no (empty context)")
        return {"needs_rewrite": "no"}

    question_lower = question.lower()

    # Hard guard 2:
    # For comparison questions, require evidence from BOTH Apple and Tesla
    # before even asking the LLM grader.
    comparison_keywords = ["compare", "comparison", "versus", "vs", "both"]
    is_comparison_question = any(word in question_lower for word in comparison_keywords)

    has_apple_source = "[Source: Apple]" in documents
    has_tesla_source = "[Source: Tesla]" in documents

    if is_comparison_question and not (has_apple_source and has_tesla_source):
        print("   Relevance Grade: no (comparison requires both Apple and Tesla evidence)")
        return {"needs_rewrite": "no"}

    system_prompt = """You are a strict financial relevance grader.

Your task is to decide whether the retrieved document context is sufficient for answering the user's question.

Rules:
1. Output ONLY one word: yes
2. Or output ONLY one word: no
3. Do not output any explanation.
4. Say yes only if the retrieved context contains useful and sufficient evidence.
5. Say no if the context is empty, unrelated, too vague, or does not support answering the question.
6. For a comparison question, say yes only if the context supports both sides of the comparison.
"""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"User question:\n{question}\n\nRetrieved document context:\n{documents}"
        )
    ]

    response = llm.invoke(msg)
    content = response.content.strip().lower()

    if content == "yes":
        grade = "yes"
    elif content == "no":
        grade = "no"
    else:
        grade = "no"

    print(f"   Relevance Grade: {grade}")
    return {"needs_rewrite": grade}

@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # Hard guard:
    # If no retrieved context exists, do not answer from model memory.
    if not documents.strip():
        return {"generation": "I don't know."}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a financial analyst.

Answer the user's question using ONLY the provided context.

Strict rules:
1. The final answer must be in English only.
2. Be very careful to distinguish 2024, 2023, and 2022 values.
3. If the provided context does not contain enough evidence to answer the question, say exactly: I don't know.
4. Do not use outside knowledge.
5. Do not guess, infer, or add extra interpretation beyond the context.
6. Keep the answer concise and direct.
7. Always cite the source in brackets, such as [Source: Apple] or [Source: Tesla].
8. If the question asks to compare two companies, but the context does not clearly support both sides of the comparison, say exactly: I don't know.

Preferred answer style:
- For a single-company factual question, answer in one sentence.
- For a comparison question, answer in one or two short sentences only if both sides are supported by the context.
- Do not explain your reasoning.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": documents,
        "question": question
    })

    answer = response.content.strip()

    # Safety check:
    # If the model gives a non-empty answer without citation and it is not exactly "I don't know.",
    # force a safer fallback.
    if answer != "I don't know." and "[Source:" not in answer:
        answer = "I don't know."

    return {"generation": answer}

@retry_logic
def rewrite_node(state: AgentState):
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()

    msg = [
        SystemMessage(
            content="""You are a financial query rewriter.

Your task is to rewrite the user's question so it becomes easier for a financial document retriever to answer.

Rules:
1. Keep the original meaning.
2. Make the question more specific and retrieval-friendly.
3. Prefer precise financial terminology when appropriate, such as:
   - revenue
   - net sales
   - operating income
   - net income
   - research and development expenses
   - capital expenditures
   - total assets
   - cash flow from operating activities
4. Preserve company names if they are mentioned.
5. Preserve years such as 2024, 2023, or 2022 if they are mentioned.
6. Do not invent new companies or new years.
7. If the question is a comparison question, rewrite it so both company names and the target financial metric are explicitly stated.
8. For Apple, "revenue" may appear as "net sales" in financial statements. Use this wording when helpful.
9. For Tesla, "revenue" may appear as "total revenues" in financial statements. Use this wording when helpful.
10. Output ONLY the rewritten question text, with no explanation.

Examples:
Original question: How much did Apple spend on new tech in 2024?
Rewritten question: What were Apple's research and development expenses in 2024?

Original question: Compare Apple and Tesla revenue in 2024.
Rewritten question: Compare Apple's net sales in 2024 with Tesla's total revenues in 2024.

Original question: Compare Apple and Tesla profit in 2024.
Rewritten question: Compare Apple's net income in 2024 with Tesla's net income in 2024.
"""
        ),
        HumanMessage(content=f"Original question: {question}")
    ]

    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

def route_after_retrieve(state: AgentState):
    """
    After the original retrieve_node stores route_target,
    decide whether to continue with the original single-path flow
    or switch into the comparison branch.
    """
    if state["route_target"] == "both":
        return "compare"
    return "single"


def decide_after_grade_apple_compare(state: AgentState):
    """
    Apple side:
    - yes -> go to Tesla retrieval
    - no  -> rewrite Apple once if under retry limit
    - if still over limit -> fail comparison
    """
    if state["apple_grade"] == "yes":
        return "tesla"

    if state["apple_search_count"] < 2:
        return "rewrite_apple"

    return "fail"


def decide_after_grade_tesla_compare(state: AgentState):
    """
    Tesla side:
    - yes -> compare generate
    - no  -> rewrite Tesla once if under retry limit
    - if still over limit -> fail comparison
    """
    if state["tesla_grade"] == "yes":
        return "compare_generate"

    if state["tesla_search_count"] < 2:
        return "rewrite_tesla"

    return "fail"

def build_graph():
    workflow = StateGraph(AgentState)

    # ===== Original single-path nodes =====
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    # ===== Comparison branch nodes =====
    workflow.add_node("prepare_compare_queries", prepare_compare_queries)

    workflow.add_node("retrieve_apple_compare", retrieve_apple_compare)
    workflow.add_node("grade_apple_compare", grade_apple_compare)
    workflow.add_node("rewrite_apple_compare", rewrite_apple_compare)

    workflow.add_node("retrieve_tesla_compare", retrieve_tesla_compare)
    workflow.add_node("grade_tesla_compare", grade_tesla_compare)
    workflow.add_node("rewrite_tesla_compare", rewrite_tesla_compare)

    workflow.add_node("compare_generate", compare_generate)
    workflow.add_node("compare_fail", compare_fail)

    # Entry point stays the same
    workflow.set_entry_point("retrieve")

    # After retrieve_node, branch by route_target
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "single": "grade_documents",
            "compare": "prepare_compare_queries",
        },
    )

    # ===== Original single-path flow =====
    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2:
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    # ===== Comparison branch flow =====
    workflow.add_edge("prepare_compare_queries", "retrieve_apple_compare")
    workflow.add_edge("retrieve_apple_compare", "grade_apple_compare")

    workflow.add_conditional_edges(
        "grade_apple_compare",
        decide_after_grade_apple_compare,
        {
            "tesla": "retrieve_tesla_compare",
            "rewrite_apple": "rewrite_apple_compare",
            "fail": "compare_fail",
        },
    )

    workflow.add_edge("rewrite_apple_compare", "retrieve_apple_compare")

    workflow.add_edge("retrieve_tesla_compare", "grade_tesla_compare")

    workflow.add_conditional_edges(
        "grade_tesla_compare",
        decide_after_grade_tesla_compare,
        {
            "compare_generate": "compare_generate",
            "rewrite_tesla": "rewrite_tesla_compare",
            "fail": "compare_fail",
        },
    )

    workflow.add_edge("rewrite_tesla_compare", "retrieve_tesla_compare")

    workflow.add_edge("compare_generate", END)
    workflow.add_edge("compare_fail", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {
        "question": question,
        "generation": "",

        # router result
        "route_target": "",

        # original single-path fields
        "documents": "",
        "search_count": 0,
        "needs_rewrite": "no",

        # comparison branch: Apple
        "apple_query": "",
        "apple_documents": "",
        "apple_grade": "no",
        "apple_search_count": 0,

        # comparison branch: Tesla
        "tesla_query": "",
        "tesla_documents": "",
        "tesla_grade": "no",
        "tesla_search_count": 0,
    }

    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    # Create one retriever tool for each company's vector database
    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(
            create_retriever_tool(
                retriever,
                f"search_{key}_financials",
                f"Search {key.capitalize()}'s financial filings and financial statement data."
            )
        )

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    # Task A ReAct prompt
    # Required placeholders:
    # {tools}, {tool_names}, {input}, {agent_scratchpad}
    template = """You are a financial analysis assistant.

You have access to the following tools:
{tools}

You must strictly follow the ReAct format below.

Rules:
1. The Final Answer must be in English only.
2. Be very careful to distinguish 2024, 2023, and 2022 values in financial tables.
3. If the exact 2024 figure is not found, say exactly: I don't know.
4. Do not guess or fabricate numbers.
5. Use tool observations as evidence before giving the final answer.

Use the following format:

Question: the input question you must answer
Thought: think carefully about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the specific query to send to the tool
Observation: the result returned by the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"
    
