#!/usr/bin/env python
# coding: utf-8

"""Paper fetching utilities for LLM research papers."""

import re
import requests
import pandas as pd
import xml.etree.ElementTree as ET


def fetch_llm_sys_papers():
    """
    Fetch LLM system papers from the LLMSys-PaperList repository.
    
    Returns:
        pd.DataFrame: DataFrame containing paper metadata with columns:
                     section, subsection, title, url
    """
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


def fetch_arxiv_abstract(urls):
    """
    Fetch abstracts from arXiv URLs.
    
    Args:
        urls (list): List of arXiv URLs
        
    Returns:
        list: List of abstracts corresponding to the input URLs
    """
    summaries = []
    for url in urls:
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if not m:
            summaries.append("")
            continue
            
        arxiv_id = m.group(1)
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        resp = requests.get(api_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            summaries.append("")
            continue

        summary = entry.find('atom:summary', ns)
        summaries.append(summary.text.strip() if summary is not None else "")
    return summaries
