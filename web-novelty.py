#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -r web_novelty_requirements.txt


# In[2]:


import json
from pathlib import Path

ENV = json.loads(Path(__file__).with_name("env.json").read_text(encoding="utf-8"))

config = {
    "dataset_dir": "./data",
    "pii_type": 'phone', #"email_cc",
    "output_dir": "./results-novelty",

    "languages": [
        "it",
        "fr",
        "sp",
        "de"
    ],

    "text_columns": {
        "it": "it",
        "fr": "fr",
        "sp": "sp",
        "de": "de"
    },

    "translation_date": "2026-01-01",

    "commoncrawl_crawls": [
        "CC-MAIN-2025-51",
        "CC-MAIN-2025-47",
        "CC-MAIN-2025-43",
        "CC-MAIN-2025-38",
        "CC-MAIN-2025-33",
        "CC-MAIN-2025-30",
        "CC-MAIN-2025-26",
        "CC-MAIN-2025-21",
        "CC-MAIN-2025-18",
        "CC-MAIN-2025-13",
        "CC-MAIN-2025-08"
    ],

    "search": {
        "provider": "tavily",
        "api_key": ENV["TAVILY_API_KEY"],
        "endpoint": "https://api.tavily.com/search",
        "max_results": 10,
        "timeout": 30
    },

    "pipeline": {
        "query_words": 50,
        "max_search_results": 10,
        "max_captures_per_url": 3,
        "commoncrawl_concurrency": 5,
        "search_delay": 1.0,
        "commoncrawl_delay": 0.2,
        "fuzzy_threshold": 90
    }
}

config["output_dir"] = f"{config['output_dir']}/{config['pii_type']}"

# In[3]:


"""
Web Novelty / Translation Provenance Pipeline

Input:
    Local Torch Face Dataset

Output:
    SQLite database
    Final CSV

Pipeline:

    Dataset
        ->
    Search engine exact phrase search
        ->
    Candidate URLs
        ->
    Common Crawl CDXJ lookup
        ->
    Historical captures
        ->
    WARC byte-range retrieval
        ->
    HTML extraction
        ->
    Exact / normalized / fuzzy matching
        ->
    Evidence classification

Designed for large experiments (~30k records).

The pipeline is resumable:
    - Every stage is persisted in SQLite.
    - Completed records are skipped.
    - API responses are cached.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import aiohttp
import pandas as pd
import requests

from bs4 import BeautifulSoup
from datasets import Dataset
from rapidfuzz.fuzz import ratio
from tqdm import tqdm


# In[4]:


# ============================================================
# DATABASE
# ============================================================

SCHEMA = """

CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    source_id INTEGER,
    language TEXT,
    text TEXT,
    translation_date TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS search_queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,

    record_id TEXT,
    query TEXT,
    provider TEXT,

    status TEXT,
    result_count INTEGER,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(record_id, query, provider)
);

CREATE TABLE IF NOT EXISTS search_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,

    query_id INTEGER,

    url TEXT,
    title TEXT,
    snippet TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(query_id, url)
);

CREATE TABLE IF NOT EXISTS commoncrawl_captures (

    capture_id INTEGER PRIMARY KEY AUTOINCREMENT,

    record_id TEXT,

    url TEXT,
    crawl TEXT,
    timestamp TEXT,

    filename TEXT,
    offset INTEGER,
    length INTEGER,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(record_id, url, crawl, timestamp)
);

CREATE TABLE IF NOT EXISTS verification (

    verification_id INTEGER PRIMARY KEY AUTOINCREMENT,

    record_id TEXT,
    url TEXT,

    crawl TEXT,
    capture_timestamp TEXT,

    exact_match INTEGER,
    normalized_match INTEGER,
    fuzzy_score REAL,

    preexisting INTEGER,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(record_id, url, crawl, capture_timestamp)
);

CREATE TABLE IF NOT EXISTS errors (

    error_id INTEGER PRIMARY KEY AUTOINCREMENT,

    record_id TEXT,
    stage TEXT,
    error TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_results_query
ON search_results(query_id);

CREATE INDEX IF NOT EXISTS idx_captures_record
ON commoncrawl_captures(record_id);

CREATE INDEX IF NOT EXISTS idx_verification_record
ON verification(record_id);

"""


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# In[5]:


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(text):

    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()


def normalize_for_matching(text):
    text = normalize_text(text)
    text = text.lower()
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()


# In[6]:


# ============================================================
# QUERY GENERATION
# ============================================================

def generate_query(
    text,
    query_words=50
):
    text = normalize_text(text)
    words = text.split()

    if len(words) <= query_words:
        return '"' + text + '"'

    # We prefer the end of document.
    # we test for memorized PII that are generated after that segment
    selected = words[-query_words:]
    query = " ".join(selected)
    return f'"{query}"'


# In[7]:


def load_records_red(
    dataset_dir,
    pii_type,
    languages,
    text_columns,
    translation_date,
    db_path,
    subset_len=150
):
    datasets ={}
    for language in languages:
        dataset_path = f"{dataset_dir}/Dataset-{pii_type}-{language}"
        datasets[language] = load_data(
            pii_type,
            dataset_path
        )

    dataset = {}
    for language in languages:
        dataset[language] = datasets[language]['context']
    dataset = Dataset.from_dict(dataset)
    
    return dataset


# In[8]:


# ============================================================
# DATASET
# ============================================================
from torch.utils.data import Subset
import torch
def sample_dataset(dataset, num_samples, seed=42):
    if num_samples > len(dataset):
        raise ValueError(
            f"num_samples ({num_samples}) cannot be larger than "
            f"dataset size ({len(dataset)})"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:num_samples]

    return Subset(dataset, indices.tolist())


def load_data(pii_type, filepath):
    """Carica e preprocessa un dataset PII."""
    data = Dataset.load_from_disk(filepath)
    data = pd.DataFrame(data)
    data['context'] = data['context'].apply(str.strip)
    # Sampling per dataset URL se troppo grande
    if len(data) > 4550 and pii_type == 'url':
        data = data.sample(n=4550, random_state=42).reset_index(drop=True)

    data = Dataset.from_pandas(data[['pii', 'context']])
    return data


def load_records(
    dataset_dir,
    pii_type,
    languages,
    text_columns,
    translation_date,
    db_path,
    subset_len=200
):
    datasets ={}
    for language in languages:
        dataset_path = f"{dataset_dir}/Dataset-{pii_type}-{language}"
        datasets[language] = load_data(
            pii_type,
            dataset_path
        )

    dataset = {}
    for language in languages:
        dataset[language] = datasets[language]['context']
    dataset = Dataset.from_dict(dataset)
    dataset = sample_dataset(dataset, subset_len, seed=42)
    dataset = Dataset.from_dict(dataset.dataset[dataset.indices])
    print("Loaded", subset_len, pii_type)
    print(dataset)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i in tqdm(
        range(len(dataset)),
        desc="Loading dataset"
    ):
        row = dataset[i]
        if i == 0:
            print(row)
        for language in languages:
            column = text_columns[language]

            text = row[column]

            if text is None:
                continue

            text = normalize_text(text)
            if not text:
                continue

            record_id = f"{i}_{language}"
            cursor.execute(
                """
                INSERT OR IGNORE INTO records
                (
                    record_id,
                    source_id,
                    language,
                    text,
                    translation_date
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    i,
                    language,
                    text,
                    translation_date
                )
            )

    conn.commit()
    conn.close()


# In[9]:


# ============================================================
# SEARCH PROVIDER
# ============================================================

class TavilySearch:

    def __init__(
        self,
        api_key,
        endpoint="https://api.tavily.com/search",
        max_results=10,
        timeout=30
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query):
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get(
            "results",
            []
        ):

            results.append({
                "url": item.get("url"),
                "title": item.get("title"),
                "snippet": item.get("content"),
            })

        return results



# In[10]:


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):
    try:
        parsed = urlparse(url)
        return urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                "",
                "",
                ""
        ))
    except Exception:
        return url



# In[11]:


# ============================================================
# COMMON CRAWL
# ============================================================

def query_commoncrawl(
    url, crawl, timeout=60):

    endpoint = (
        f"https://index.commoncrawl.org/"
        f"{crawl}-index"
    )

    params = {"url": url, "output": "json", "filter": "status:200", "collapse": "digest"}

    response = requests.get(endpoint, params=params,timeout=timeout)

    # 404 means URL not found
    if response.status_code == 404:
        return []

    response.raise_for_status()
    captures = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        captures.append(data)

    return captures


# In[12]:


# ============================================================
# WARC DOWNLOAD
# ============================================================

async def fetch_warc_record(
    session,
    filename,
    offset,
    length
):

    url = (
        "https://data.commoncrawl.org/"
        + filename
    )

    end = offset + length - 1

    headers = {"Range":f"bytes={offset}-{end}"}

    async with session.get(url,
        headers=headers,
        timeout=aiohttp.ClientTimeout(
            total=120
        )
    ) as response:

        response.raise_for_status()
        return await response.read()


# ============================================================
# WARC HTML EXTRACTION
# ============================================================

def extract_html_from_warc(
    compressed_data
):

    try:
        data = gzip.decompress(
            compressed_data
        )
    except Exception:

        data = compressed_data

    # WARC record contains headers
    # followed by HTTP response.

    separator = b"\r\n\r\n"

    parts = data.split(
        separator,
        2
    )

    if len(parts) < 3:
        return ""

    http_response = parts[2]
    # Separate HTTP headers
    # from HTML body.
    parts = http_response.split(
        separator,
        1
    )

    if len(parts) != 2:
        return ""

    body = parts[1]
    try:
        html = body.decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:
        return ""
    return html


def html_to_text(html):
    if not html:
        return ""

    soup = BeautifulSoup(html,"lxml")
    for element in soup(["script","style","noscript","svg"]):
        element.decompose()
    text = soup.get_text(separator=" ")

    return normalize_text(text)


# In[13]:


# ============================================================
# MATCHING
# ============================================================

def verify_match(
    source_text,
    web_text,
    fuzzy_threshold=90
):

    source_normalized = (
        normalize_for_matching(
            source_text
        )
    )

    web_normalized = (
        normalize_for_matching(
            web_text
        )
    )

    exact_match = (
        source_text in web_text
    )

    normalized_match = (
        source_normalized
        in
        web_normalized
    )

    # For long passages, compare
    # chunks rather than entire pages.

    fuzzy_score = 0.0

    if source_normalized:
        source_words = (
            source_normalized
            .split()
        )

        window_size = len(
            source_words
        )

        web_words = (
            web_normalized
            .split()
        )

        if len(web_words) <= window_size:
            fuzzy_score = ratio(
                source_normalized,
                web_normalized
            )
        else:
            # Check a limited number
            # of windows.
            step = max(
                1,
                window_size // 4
            )
            best = 0
            for start in range(
                0,
                len(web_words)
                - window_size
                + 1,
                step
            ):

                window = " ".join(
                    web_words[
                        start:
                        start
                        + window_size
                    ]
                )

                score = ratio(
                    source_normalized,
                    window
                )

                best = max(
                    best,
                    score
                )

                if best >= 99:
                    break

            fuzzy_score = best

    fuzzy_match = (
        fuzzy_score
        >= fuzzy_threshold
    )

    return {
        "exact_match":
            int(exact_match),

        "normalized_match":
            int(normalized_match),

        "fuzzy_score":
            fuzzy_score,

        "fuzzy_match":
            int(fuzzy_match)
    }


# In[14]:


# ============================================================
# SEARCH PROCESSING
# ============================================================

def process_searches(
    db_path,
    search_client,
    query_words,
    max_results,
    search_delay
):

    conn = sqlite3.connect(
        db_path
    )

    records = conn.execute(

        """
        SELECT
            record_id,
            text
        FROM records
        WHERE record_id NOT IN
        (
            SELECT DISTINCT
                record_id
            FROM search_queries
            WHERE status = 'completed'
        )
        """
    ).fetchall()

    for record_id, text in tqdm(
        records,
        desc="Searching web"
    ):

        try:

            query = generate_query(
                text,
                query_words
            )
            if record_id.startswith("0_"):
                print(query)
            cursor = conn.cursor()

            cursor.execute(

                """
                INSERT OR IGNORE INTO
                search_queries
                (
                    record_id,
                    query,
                    provider,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,

                (
                    record_id,
                    query,
                    "tavily",
                    "running"
                )
            )

            conn.commit()

            query_id = cursor.execute(

                """
                SELECT query_id
                FROM search_queries
                WHERE record_id = ?
                AND query = ?
                """,

                (
                    record_id,
                    query
                )
            ).fetchone()[0]

            results = search_client.search(

                query
            )

            for result in results:

                url = normalize_url(

                    result["url"]
                )

                cursor.execute(

                    """
                    INSERT OR IGNORE INTO
                    search_results
                    (
                        query_id,
                        url,
                        title,
                        snippet
                    )
                    VALUES (?, ?, ?, ?)
                    """,

                    (
                        query_id,
                        url,
                        result["title"],
                        result["snippet"]
                    )
                )

            cursor.execute(

                """
                UPDATE search_queries
                SET
                    status = 'completed',
                    result_count = ?
                WHERE query_id = ?
                """,

                (
                    len(results),
                    query_id
                )
            )

            conn.commit()

            time.sleep(
                search_delay
            )

        except Exception as e:

            cursor.execute(

                """
                UPDATE search_queries
                SET status = 'error'
                WHERE query_id = ?
                """,

                (
                    query_id,
                )
            )

            cursor.execute(

                """
                INSERT INTO errors
                (
                    record_id,
                    stage,
                    error
                )
                VALUES (?, ?, ?)
                """,

                (
                    record_id,
                    "search",
                    str(e)
                )
            )

            conn.commit()


    conn.close()


# In[15]:


# ============================================================
# COMMON CRAWL INDEX PROCESSING
# ============================================================

def process_commoncrawl_index(
    db_path,
    crawls,
    translation_date,
    delay=0.2

):

    conn = sqlite3.connect(
        db_path
    )

    records = conn.execute(
        """
        SELECT DISTINCT
            r.record_id,
            r.text,
            sr.url
        FROM records r

        JOIN search_queries sq
        ON r.record_id = sq.record_id

        JOIN search_results sr
        ON sq.query_id = sr.query_id

        WHERE r.record_id IN
        (
            SELECT DISTINCT
                record_id
            FROM search_queries
            WHERE status = 'completed'
        )
        """
    ).fetchall()

    for (
        record_id,
        text,
        url
    ) in tqdm(
        records,
        desc="Common Crawl index"
    ):

        for crawl in crawls:
            try:
                captures = (
                    query_commoncrawl(
                        url,
                        crawl
                    )
                )

                for capture in captures:
                    timestamp = (
                        capture.get(
                            "timestamp"
                        )
                    )

                    if not timestamp:
                        continue

                    # YYYYMMDDhhmmss

                    if timestamp >= (translation_date.replace("-", "")):
                        continue

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO
                        commoncrawl_captures
                        (
                            record_id,
                            url,
                            crawl,
                            timestamp,
                            filename,
                            offset,
                            length
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id,
                            url,
                            crawl,
                            timestamp,
                            capture.get("filename"),
                            capture.get("offset"),
                            capture.get("length")
                        )
                    )

                conn.commit()

                time.sleep(delay)

            except Exception as e:
                conn.execute(
                    """
                    INSERT INTO errors
                    (
                        record_id,
                        stage,
                        error
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        record_id,
                        "commoncrawl_index",
                        str(e)
                    )
                )
                conn.commit()


    conn.close()


# In[16]:


# ============================================================
# WARC VERIFICATION
# ============================================================

async def verify_captures(
    db_path,
    fuzzy_threshold=90,
    concurrency=5

):

    conn = sqlite3.connect(
        db_path
    )

    rows = conn.execute(
        """
        SELECT
        c.capture_id,
        c.record_id,
        c.url,
        c.crawl,
        c.timestamp,
        c.filename,
        c.offset,
        c.length,
        r.text
        FROM commoncrawl_captures c
        JOIN records r
        ON c.record_id = r.record_id
        WHERE c.capture_id NOT IN
        (SELECT verification_id FROM verification)
        """
    ).fetchall()

    conn.close()


    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def process(row):
            (capture_id,
             record_id,
             url,
             crawl,
             timestamp,
             filename,
             offset,
             length,
             source_text
            ) = row


            async with semaphore:
                try:
                    data = await fetch_warc_record(
                        session,
                        filename,
                        int(offset),
                        int(length)
                    )

                    html = (
                        extract_html_from_warc(
                            data
                        )
                    )

                    web_text = (
                        html_to_text(
                            html
                        )
                    )

                    result = verify_match(
                        source_text,
                        web_text,
                        fuzzy_threshold
                    )

                    return (capture_id,
                        record_id,
                        url,
                        crawl,
                        timestamp,
                        result,
                        None
                    )

                except Exception as e:
                    return (capture_id,
                        record_id,
                        url,
                        crawl,
                        timestamp,
                        None,
                        str(e)
                    )


        tasks = [process(row) for row in rows]

        results = []
        for future in tqdm(asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Verifying WARC"
        ):

            result = await future
            results.append(
                result
            )


    conn = sqlite3.connect(
        db_path
    )

    for result in results:
        (capture_id,
         record_id,
         url,
         crawl,
         timestamp,
         match,
         error
         ) = result


        if error:
            conn.execute(
                """
                INSERT INTO errors
                (
                    record_id,
                    stage,
                    error
                )
                VALUES (?, ?, ?)
                """,
                (
                    record_id,
                    "warc",
                    error
                )
            )

            continue


        preexisting = int(
            match["exact_match"]
            or
            match["normalized_match"]
            or
            match["fuzzy_match"]
        )


        conn.execute(
            """
            INSERT OR IGNORE INTO
            verification
            (
                record_id,
                url,
                crawl,
                capture_timestamp,
                exact_match,
                normalized_match,
                fuzzy_score,
                preexisting
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                record_id,
                url,
                crawl,
                timestamp,
                match["exact_match"],
                match["normalized_match"],
                match["fuzzy_score"],
                preexisting
            )
        )


    conn.commit()
    conn.close()


# In[17]:


# ============================================================
# FINAL CSV
# ============================================================

def export_results(
    db_path,
    output_csv

):

    conn = sqlite3.connect(
        db_path
    )


    query = """
    WITH search_summary AS (
        SELECT
            record_id,
            COUNT(*) AS queries,
            SUM(
                CASE
                WHEN result_count > 0
                THEN 1
                ELSE 0
                END
            ) AS queries_with_results
        FROM search_queries
        GROUP BY record_id

    ),

    capture_summary AS (
        SELECT
            record_id,
            COUNT(*) AS historical_captures,
            MIN(timestamp)
                AS earliest_capture
        FROM commoncrawl_captures
        GROUP BY record_id

    ),

    verification_summary AS (
        SELECT
            record_id,
            MAX(exact_match)
                AS exact_match,

            MAX(normalized_match)
                AS normalized_match,

            MAX(fuzzy_score)
                AS max_fuzzy_score,

            MAX(preexisting)
                AS preexisting

        FROM verification
        GROUP BY record_id
    )

    SELECT
        r.record_id,
        r.source_id,
        r.language,
        r.translation_date,
        ss.queries,
        ss.queries_with_results,

        COALESCE(
            cs.historical_captures,
            0
        ) AS historical_captures,

        cs.earliest_capture,

        COALESCE(
            vs.exact_match,
            0
        ) AS exact_match,

        COALESCE(
            vs.normalized_match,
            0
        ) AS normalized_match,

        COALESCE(
            vs.max_fuzzy_score,
            0
        ) AS max_fuzzy_score,

        COALESCE(
            vs.preexisting,
            0
        ) AS preexisting,

        CASE
            WHEN COALESCE(
                vs.preexisting,
                0
            ) = 1

            THEN 'PREEXISTING_TEXT_MATCH'
            WHEN COALESCE(
                cs.historical_captures,
                0
            ) > 0

            THEN 'HISTORICAL_URL_NO_TEXT_MATCH'
            WHEN COALESCE(
                ss.queries_with_results,
                0
            ) > 0

            THEN 'SEARCH_MATCH_NO_PREHISTORY'
            ELSE 'SEARCH_NO_MATCH'
        END AS evidence_level

    FROM records r
    LEFT JOIN search_summary ss
    ON r.record_id = ss.record_id
    LEFT JOIN capture_summary cs
    ON r.record_id = cs.record_id
    LEFT JOIN verification_summary vs
    ON r.record_id = vs.record_id
    ORDER BY
        r.source_id,
        r.language

    """


    df = pd.read_sql_query(
        query,
        conn
    )


    df.to_csv(
        output_csv,
        index=False
    )


    conn.close()

    return df



# In[18]:


config


# In[19]:


stage = "all"

#"load", "search", "commoncrawl", "verify", "export", "all"
output_dir = Path(config["output_dir"])

output_dir.mkdir(parents=True, exist_ok=True)

db_path = (
    output_dir
    /
    "novelty.db"
)


csv_path = (
    output_dir
    /
    "novelty_results.csv"
)


init_db(db_path)


# In[20]:


# --------------------------------------------------------
# LOAD
# --------------------------------------------------------

if stage in [
    "load",
    "all"
]:

    load_records(
        config["dataset_dir"],
        config["pii_type"],
        config["languages"],
        config["text_columns"],
        config["translation_date"],
        db_path
    )


# In[22]:


# --------------------------------------------------------
# SEARCH
# --------------------------------------------------------

if stage in [
    "search",
    "all"
]:

    search_config = config["search"]

    client = TavilySearch(
        api_key=search_config["api_key"],
        endpoint=search_config["endpoint"],
        max_results=search_config["max_results"],
        timeout=search_config["timeout"]
    )


    pipeline_config = config["pipeline"]


    process_searches(
        db_path,
        client,
        pipeline_config["query_words"],
        pipeline_config["max_search_results"],
        pipeline_config["search_delay"]
    )


# In[23]:


# --------------------------------------------------------
# COMMON CRAWL INDEX
# --------------------------------------------------------

if stage in [
    "commoncrawl",
    "all"

]:

    process_commoncrawl_index(
        db_path,
        config[
            "commoncrawl_crawls"
        ],
        config[
            "translation_date"
        ],
        config[
            "pipeline"
        ][
            "commoncrawl_delay"
        ]
    )


# In[24]:


# --------------------------------------------------------
# VERIFY WARC
# --------------------------------------------------------

if stage in [
    "verify",
    "all"
]:

    asyncio.run(
        verify_captures(
            db_path,
            config[
                "pipeline"
            ][
                "fuzzy_threshold"
            ],

            config[
                "pipeline"
            ][
                "commoncrawl_concurrency"
            ]
        )
    )


# In[ ]:


# --------------------------------------------------------
# EXPORT
# --------------------------------------------------------

if stage in [
    "export",
    "all"
]:

    df = export_results(

        db_path,

        csv_path
    )


    print()

    print(

        "Finished."
    )

    print(

        f"SQLite: {db_path}"
    )

    print(

        f"CSV:    {csv_path}"
    )

    print()

    print(

        df[
            "evidence_level"
        ].value_counts()
    )

