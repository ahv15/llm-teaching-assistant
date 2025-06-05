#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import re
import pandas as pd
def fetch_llm_sys_papers():
    url = "https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md"
    response = requests.get(url)
    lines = response.text.splitlines()
    section_pattern = re.compile(r'^##\s+(.*)')
    subsection_pattern = re.compile(r'^###\s+(.*)')
    link_pattern = re.compile(r'- \[(.*?)\]\((.*?)\)')

    entries = []
    current_section = None
    current_subsection = None

    for line in lines:
        line = line.strip()
        sec_match = section_pattern.match(line)
        if sec_match:
            current_section = sec_match.group(1)
            current_subsection = None
            continue
        sub_match = subsection_pattern.match(line)
        if sub_match:
            current_subsection = sub_match.group(1)
            continue
        link_match = link_pattern.match(line)
        if link_match and current_section:
            title, url = link_match.groups()
            if url.startswith("http"):
                entries.append({
                    "section": current_section,
                    "subsection": current_subsection or "",
                    "title": title,
                    "url": url
                })

    df = pd.DataFrame(entries)
    return df


# In[ ]:


df = fetch_llm_sys_papers()
arxiv_urls = df.loc[df['url'].str.contains(r'arxiv\.org/(abs|pdf)', na=False), 'url'].tolist()


# In[4]:


import xml.etree.ElementTree as ET
def fetch_arxiv_abstract(urls):
    summaries = []
    for url in urls:
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if not m:
            return ""
        arxiv_id = m.group(1)
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        resp = requests.get(api_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            return ""

        summary = entry.find('atom:summary', ns)
        summaries.append(summary.text.strip() if summary is not None else "")
    return summaries


# In[5]:


summaries = fetch_arxiv_abstract(arxiv_urls)


# In[6]:


import openai
import faiss
import numpy as np
from langgraph.graph import StateGraph, START, END


# In[7]:


emb_list = []
for summary in summaries:
    resp = openai.embeddings.create(input=summary, model="text-embedding-3-small")
    emb_list.append(resp.data[0].embedding)

emb_array = np.array(emb_list, dtype="float32")

d = emb_array.shape[1]
index = faiss.IndexFlatL2(d)
index.add(emb_array)
faiss.write_index(index, "summary.faiss")

index = faiss.read_index("summary.faiss")


# In[8]:


import json
with open("urls.json", "w", encoding="utf-8") as f:
    json.dump(arxiv_urls, f, ensure_ascii=False, indent=2)


# In[9]:


index = faiss.read_index("summary.faiss")


# In[10]:


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


# In[14]:


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


# In[15]:


model = init_chat_model("openai:gpt-4o")
summarization_model = model.bind(max_tokens=128)


# In[16]:


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)


# In[29]:


def react_with_tools(state: LLMInputState) -> dict[str, Any]:
    ctx = state.get("context", {})
    chat_history: List[AnyMessage] = state["summarized_messages"]
    running_summary = state["context"].get("running_summary", "")
    last_user_msg = ""
    for msg in reversed(chat_history):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content

    tools_block = "paper_retriever"

    prompt_text = f"""
        Temporary Prompt
    """
    response = model.invoke([{ "role": "user", "content": prompt_text }])
    raw_text: str = response.content

    action_pattern = r"Action:\s*paper_retriever\s*[\r\n]+Action Input:\s*(.+)"
    m = re.search(action_pattern, raw_text)
    if m:
        paper_query = m.group(1).strip()
        tool_output = paper_retriever(paper_query)
        final_message = AIMessage(content=tool_output)

    else:
        final_message = response

    return {
        "messages": [final_message],
        "context": ctx
    }


# In[ ]:


checkpointer = InMemorySaver()
builder = StateGraph(State)

builder.add_node("summarize", summarization_node)
builder.add_node("react_with_tools", react_with_tools)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "react_with_tools")
graph = builder.compile(checkpointer=checkpointer)

config = { "configurable": { "thread_id": "1" } }

first_turn = graph.invoke(
    { 
      "messages": "Can you teach me about some new LLM Training optimizations?",
      "context": {} 
    },
    config
)
first_turn["messages"][-1].pretty_print()

