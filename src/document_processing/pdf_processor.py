#!/usr/bin/env python
# coding: utf-8

"""PDF processing and section extraction using GROBID."""

import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict


class PDFProcessor:
    """Handles PDF processing using GROBID service."""
    
    def __init__(self, grobid_url: str = 'http://localhost:8070'):
        """
        Initialize PDF processor.
        
        Args:
            grobid_url (str): URL of the GROBID service
        """
        self.grobid_url = grobid_url
    
    def fetch_and_extract_sections(self, arxiv_url: str) -> Dict[str, str]:
        """
        Fetch PDF from arXiv and extract sections using GROBID.
        
        Args:
            arxiv_url (str): arXiv URL (abs or pdf)
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to content
        """
        # Convert abs URL to PDF URL if needed
        if '/abs/' in arxiv_url:
            pid = arxiv_url.rstrip('/').split('/')[-1]
            pdf_url = f'https://arxiv.org/pdf/{pid}.pdf'
        else:
            pdf_url = arxiv_url

        # Fetch PDF
        r = requests.get(pdf_url)
        r.raise_for_status()
        pdf_bytes = r.content

        # Send to GROBID
        files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
        r = requests.post(f'{self.grobid_url}/api/processFulltextDocument', files=files)
        r.raise_for_status()
        tei = r.text

        # Parse TEI
        TEI_NS = 'http://www.tei-c.org/ns/1.0'
        ET.register_namespace('tei', TEI_NS)
        root = ET.fromstring(tei)

        sections: Dict[str, str] = {}
        for div in root.findall(f'.//{{{TEI_NS}}}div'):
            if div.attrib.get('type') == 'body':
                continue

            sec_key = (
                div.attrib.get('type')
                or div.attrib.get('subtype')
                or next((h.text for h in div.findall(f'.//{{{TEI_NS}}}head') if h.text), None)
            )
            if not sec_key:
                continue

            parts = []
            for el in div.iter():
                if el.text and el.tag != f'{{{TEI_NS}}}head':
                    parts.append(el.text.strip())
                if el.tail:
                    parts.append(el.tail.strip())

            sections[sec_key.lower()] = ' '.join(p for p in parts if p)

        return sections
