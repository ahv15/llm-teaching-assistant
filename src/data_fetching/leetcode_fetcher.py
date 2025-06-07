"""
LeetCode Data Fetching

This module provides tools for fetching LeetCode problems and integrating
with the teaching assistant for coding interview practice.
"""

import random
import requests
import bs4
import time
from typing import Optional, Set, Dict, List

from langchain.tools import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Constants
CATALOG_URL = "https://leetcode.com/api/problems/algorithms/"
GRAPHQL_ENDPOINT = "https://leetcode.com/graphql"
GRAPHQL_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) { content }
}
"""


def get_catalog() -> List[Dict]:
    """
    Fetch the LeetCode problems catalog.
    
    Returns:
        List of problem dictionaries with metadata
    """
    response = requests.get(CATALOG_URL, timeout=15)
    data = response.json()
    
    difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}
    
    return [
        {
            "slug": problem["stat"]["question__title_slug"],
            "title": problem["stat"]["question__title"],
            "difficulty": difficulty_map[problem["difficulty"]["level"]],
            "paid_only": problem["paid_only"],
        }
        for problem in data["stat_status_pairs"]
    ]


def pick_random_problem(
    catalog: List[Dict], 
    *, 
    allow_premium: bool = False, 
    difficulties: Optional[Set[str]] = None
) -> Optional[Dict]:
    """
    Pick a random problem from the catalog based on criteria.
    
    Args:
        catalog: List of problems from get_catalog()
        allow_premium: Whether to include premium problems
        difficulties: Set of allowed difficulties ("Easy", "Medium", "Hard")
        
    Returns:
        Random problem dictionary or None if no matches
    """
    filtered_problems = [
        problem for problem in catalog
        if (allow_premium or not problem["paid_only"])
        and (difficulties is None or problem["difficulty"] in difficulties)
    ]
    
    return random.choice(filtered_problems) if filtered_problems else None


def fetch_statement(slug: str) -> str:
    """
    Fetch the problem statement for a given LeetCode problem slug.
    
    Args:
        slug: The problem slug (e.g., "two-sum")
        
    Returns:
        Clean text version of the problem statement
    """
    response = requests.post(
        GRAPHQL_ENDPOINT,
        json={"query": GRAPHQL_QUERY, "variables": {"titleSlug": slug}},
        timeout=15,
    )
    
    result = response.json()
    html_content = result["data"]["question"]["content"]
    
    # Parse HTML and extract clean text
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text("\n")
    
    return clean_text.strip()


@tool(
    description="Fetch a random LeetCode problem statement for interview practice.",
    return_direct=True
)
def get_problem() -> Dict:
    """
    Fetch a random LeetCode problem for coding practice.
    
    This tool fetches a random Medium or Hard difficulty LeetCode problem
    by calling get_catalog(), pick_random_problem(), and fetch_statement().
    
    Returns:
        Dictionary containing problem title, difficulty, slug, and statement
    """
    catalog = get_catalog()
    problem = pick_random_problem(catalog, difficulties={"Medium", "Hard"})
    
    if not problem:
        return {"error": "No suitable problems found"}
    
    statement_text = fetch_statement(problem["slug"])
    
    return {
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "slug": problem["slug"],
        "statement": statement_text
    }


class SeleniumLeetCodeFetcher:
    """
    Alternative LeetCode fetcher using Selenium for dynamic content.
    
    This class provides an alternative method for fetching LeetCode problems
    when the API-based approach doesn't work or for problems that require
    browser interaction.
    """
    
    def __init__(self):
        """Initialize the Selenium-based fetcher."""
        self.seen_problems: Set[str] = set()
    
    def get_random_problem_selenium(self) -> tuple[str, str]:
        """
        Get a random problem using Selenium browser automation.
        
        Returns:
            Tuple of (title, url) for the problem
        """
        driver = webdriver.Chrome()
        driver.get(
            "https://takeuforward.org/interviews/strivers-sde-sheet-top-coding-interview-problems"
        )
        wait = WebDriverWait(driver, 10)

        def get_current_problem():
            """Helper to read the current top-row problem."""
            anchor = wait.until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "tbody tr:first-child td:first-child a[href]"
                ))
            )
            return anchor.text.strip(), anchor.get_attribute("href")

        def click_pick_random():
            """Helper to click 'Pick Random' and wait for table update."""
            button = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[.//span[normalize-space()='Pick Random']]"
                ))
            )
            time.sleep(5)
            button.click()
            # Wait until the title in the top row changes to something new & non-empty
            wait.until(lambda _: get_current_problem()[0] not in ("", "-"))
            return get_current_problem()

        # Loop until we get an unseen problem
        while True:
            title, url = click_pick_random()
            if title not in self.seen_problems:
                self.seen_problems.add(title)
                break

        driver.quit()
        return title, url
    
    def get_problem_statement_selenium(self) -> str:
        """
        Get a complete problem statement using Selenium.
        
        Returns:
            The problem statement text
        """
        title, url = self.get_random_problem_selenium()
        
        # Load the problem page
        driver = webdriver.Chrome()
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        time.sleep(5)
        
        # Wait for the problem text container
        css_selector = (
            "div.article.mt-2.font-medium.text-zinc-800."
            "dark\\:text-zinc-200.font-dmSans"
        )
        target = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))

        # Expand it if needed
        try:
            header = target.find_element(
                By.XPATH,
                "./ancestor::div[contains(@class,'wp-block-heading')]"
            )
            if header.get_attribute("aria-expanded") == "false":
                driver.execute_script("arguments[0].click();", header)
                wait.until(lambda d: header.get_attribute("aria-expanded") == "true")
        except Exception:
            pass

        problem_text = target.text.strip()
        driver.quit()
        
        return problem_text


# Example usage and testing
if __name__ == "__main__":
    # Test the API-based approach
    catalog = get_catalog()
    problem = pick_random_problem(catalog, difficulties={"Medium", "Hard"})
    
    if problem:
        print(f"{problem['title']} ({problem['difficulty']}) → {problem['slug']}\n")
        statement = fetch_statement(problem["slug"])
        print(statement[:800], "…")
