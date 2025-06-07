import os
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor


# In[11]:


@tool(description="Return the teaching content of the paper closest to the query.", return_direct=True)
def paper_retriever(query: str) -> str:
    """
    Given a query string, searches a FAISS index over arXiv paper embeddings
    and returns the teaching content of the best-matched paper.

    Parameters
    ----------
    query : str
        A natural-language query to find the most relevant paper.

    Returns
    -------
    str
        The teaching content of the retrieved arXiv paper.
    """
    import json
    from io import BytesIO
    import numpy as np
    import faiss
    import requests
    import xml.etree.ElementTree as ET
    from openai import OpenAI

    def generate_section_lesson(
        section_name: str,
        section_text: str,
        next_section_name: str | None = None,
    ) -> str:
        # Base instruction
        prompt = f"""
        Youre an expert teaching a research paper.
        Please turn the following section "{section_name}" into a beginner-friendly lesson fragment:

        {section_text}

        Your fragment should:
          - Explain every key idea clearly with any math worked out step by step if needed.
          - Use examples wherever helpful.
        """

        # If we know the next section, ask for a transition
        if next_section_name:
            prompt += (
                f'\nAt the end, include one sentence that smoothly '
                f'transitions the learner into the next section, "{next_section_name}".'
            )

        prompt += "\n\nLesson fragment:\n"
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def fetch_and_grobid_sections(arxiv_url: str,
                                  grobid_url: str = 'http://localhost:8070'
    ) -> dict[str, str]:
        if '/abs/' in arxiv_url:
            pid = arxiv_url.rstrip('/').split('/')[-1]
            pdf_url = f'https://arxiv.org/pdf/{pid}.pdf'
        else:
            pdf_url = arxiv_url

        # 2) Fetch PDF
        r = requests.get(pdf_url)
        r.raise_for_status()
        pdf_bytes = r.content

        # 3) Send to GROBID
        files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
        r = requests.post(f'{grobid_url}/api/processFulltextDocument', files=files)
        r.raise_for_status()
        tei = r.text

        # 4) Parse TEI
        TEI_NS = 'http://www.tei-c.org/ns/1.0'
        ET.register_namespace('tei', TEI_NS)
        root = ET.fromstring(tei)

        sections: dict[str, str] = {}
        for div in root.findall(f'.//{{{TEI_NS}}}div'):
            if div.attrib.get('type') == 'body':
                continue

            sec_key = (
                div.attrib.get('type')
                or div.attrib.get('subtype')
                or next((h.text for h in div.findall(f'./{{{TEI_NS}}}head') if h.text), None)
            )
            if not sec_key:
                continue

            parts: list[str] = []
            for el in div.iter():
                if el.text and el.tag != f'{{{TEI_NS}}}head':
                    parts.append(el.text.strip())
                if el.tail:
                    parts.append(el.tail.strip())

            sections[sec_key.lower()] = ' '.join(p for p in parts if p)

        return sections

    # 5) Embed the query and retrieve best URL
    resp = OpenAI().embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    q_emb = np.array([resp.data[0].embedding], dtype='float32')
    index = faiss.read_index('summary.faiss')
    _, I = index.search(q_emb, k=1)
    best_idx = int(I[0, 0])

    with open('urls.json', 'r', encoding='utf-8') as f:
        urls = json.load(f)
    best_url = urls[best_idx]

    # 6) Fetch, section, summarize
    sections = fetch_and_grobid_sections(best_url)
    full_lessons = []
    names = list(sections.keys())
    for i, sec in enumerate(names):
        text = sections[sec]
        nxt = names[i+1] if i+1 < len(names) else None
        frag = generate_section_lesson(sec, text, next_section_name=nxt)
        full_lessons.append(f"## {sec.title()}\n\n{frag}")

    complete_course = "\n\n".join(full_lessons)

    return complete_course


# In[12]:


from typing import Any, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

class State(MessagesState):
    context: dict[str, Any]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


# In[13]:


model = init_chat_model("openai:gpt-4o")
summarization_model = model.bind(max_tokens=128)


# In[14]:


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)


# In[15]:


from leetcodelib import get_catalog, pick_random_problem, fetch_statement


# In[136]:


@tool(
  description="Fetch a random LeetCode problem statement for interview practice.",
  return_direct=True
)
def get_problem() -> dict:
    """
    Wrapper that fetches a random LeetCode problem 
    by calling get_catalog, pick_random_problem, and fetch_statement.
    """
    catalog = get_catalog()
    prob    = pick_random_problem(catalog, difficulties={"Medium", "Hard"})
    statement_text = fetch_statement(prob["slug"])
    return {
        "title": prob["title"],
        "difficulty": prob["difficulty"],
        "slug": prob["slug"],
        "statement": statement_text
    }


# In[187]:


def route_after_tools(state: dict[str, Any]) -> str:
    msgs = state["messages"]
    if not msgs:
        return "agent"
    last_msg = msgs[-1]
    if isinstance(last_msg, ToolMessage) and last_msg.name == "paper_retriever":
        return "end"
    return "agent"


# In[188]:


def route_after_agent(state: dict[str, Any]) -> str:
    last = state["messages"][-1]
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "get_problem":
            return "end"
    if last.additional_kwargs.get("tool_calls"):
        return "tools"
    return "end"


# In[189]:


model = init_chat_model("openai:gpt-4o").bind_tools([paper_retriever, get_problem])
summarization_model = model.bind(max_tokens=128)


# In[190]:


from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, AIMessage
tool_node = ToolNode([paper_retriever, get_problem])


# In[191]:


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)


# In[192]:


def agent_node(state: MessagesState) -> dict[str, Any]:
    """
    This node takes prior messages in state["messages"], sends them 
    to the LLM, and returns the new AIMessage. If the LLM suggests a tool_call, 
    that will be in ai_message.tool_calls.
    """
    last_human_content = ""
    last_tool_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_content = msg.content
            break
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            last_tool_content = msg.content
            break

    response = model.invoke(last_human_content + last_tool_content)
    print(response)
    return {"messages": state["messages"] + [response]}


# In[193]:


checkpointer = InMemorySaver()
builder = StateGraph(State)

builder.add_node("summarize", summarization_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START,       "summarize")
builder.add_edge("summarize", "agent")

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
      "tools": "tools",
      "end":   END
    }
)

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent":    "agent",
        "end":       END
    }
)

graph = builder.compile(checkpointer=checkpointer)


# In[185]:


from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

