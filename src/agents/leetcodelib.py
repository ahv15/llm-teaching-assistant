#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui  import WebDriverWait
from selenium.webdriver.support      import expected_conditions as EC
import time


# In[2]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def getRandomProblem(seen):
    # 1) launch browser & navigate
    driver = webdriver.Chrome()
    driver.get(
        "https://takeuforward.org/interviews/strivers-sde-sheet-top-coding-interview-problems"
    )
    wait = WebDriverWait(driver, 10)

    # 2) helper to read the current top-row problem
    def get_current_problem():
        anchor = wait.until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "tbody tr:first-child td:first-child a[href]"
            ))
        )
        return anchor.text.strip(), anchor.get_attribute("href")

    # 3) helper to click “Pick Random” and wait for the table to update
    def click_pick_random():
        btn = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[.//span[normalize-space()='Pick Random']]"
            ))
        )
        time.sleep(5)
        btn.click()
        # wait until the title in the top row changes to something new & non-empty
        wait.until(lambda _: get_current_problem()[0] not in ("", "-"))
        return get_current_problem()

    # 4) loop until we get an unseen problem
    while True:
        title, url = click_pick_random()
        if title not in seen:
            seen.add(title)
            break

    driver.quit()
    return title, url

def getProblem():
    seen = set()
    title, url = getRandomProblem(seen)
    # now load the problem page...
    driver = webdriver.Chrome()
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    time.sleep(5)
    # wait for the problem text container
    css_sel = (
        "div.article.mt-2.font-medium.text-zinc-800."
        "dark\\:text-zinc-200.font-dmSans"
    )
    target = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css_sel)))

    # Expand it if needed
    try:
        hdr = target.find_element(
            By.XPATH,
            "./ancestor::div[contains(@class,'wp-block-heading')]"
        )
        if hdr.get_attribute("aria-expanded") == "false":
            driver.execute_script("arguments[0].click();", hdr)
            wait.until(lambda d: hdr.get_attribute("aria-expanded") == "true")
    except Exception:
        pass

    problem = target.text.strip()
    driver.quit()
    return problem

# Usage
problem_text = getProblem()
print(problem_text)


# In[4]:


import random, requests, bs4, pprint

CATALOG_URL = "https://leetcode.com/api/problems/algorithms/"

def get_catalog():
    data = requests.get(CATALOG_URL, timeout=15).json()
    diffs = {1: "Easy", 2: "Medium", 3: "Hard"}
    return [
        {
            "slug":      s["stat"]["question__title_slug"],
            "title":     s["stat"]["question__title"],
            "difficulty": diffs[s["difficulty"]["level"]],
            "paid_only": s["paid_only"],
        }
        for s in data["stat_status_pairs"]
    ]


def pick_random_problem(catalog, *, allow_premium=False, difficulties=None):
    pool = [
        p for p in catalog
        if (allow_premium or not p["paid_only"])
        and (difficulties is None or p["difficulty"] in difficulties)
    ]
    return random.choice(pool) if pool else None


GQL = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) { content }
}
"""

def fetch_statement(slug):
    resp = requests.post(
        "https://leetcode.com/graphql",
        json={"query": GQL, "variables": {"titleSlug": slug}},
        timeout=15,
    ).json()

    html  = resp["data"]["question"]["content"]
    text  = bs4.BeautifulSoup(html, "html.parser").get_text("\n")
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    catalog = get_catalog()
    prob    = pick_random_problem(catalog, difficulties={"Medium", "Hard"})
    print(f"{prob['title']}  ({prob['difficulty']})  →  {prob['slug']}\n")

    statement = fetch_statement(prob["slug"])
    print(statement[:800], "…")

