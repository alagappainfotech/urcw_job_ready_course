# Module 7: Data Handling, Persistence & Exploration

## Learning Objectives
- Perform ETL (extract, transform, load) from files, web, and APIs
- Clean, normalize, and validate data
- Persist data with files and databases (SQLite + concepts for Postgres)
- Explore and visualize data (pandas + matplotlib overview)

## Core Concepts
- File wrangling: pathlib, shutil, zipfile, tarfile
- Delimited data (csv), Excel (openpyxl), JSON (json), XML (xml / xmltodict)
- HTTP/HTTPS (requests), APIs (REST/JSON), scraping basics (BeautifulSoup)
- Databases: DB-API (sqlite3), ORM overview (SQLAlchemy)
- Pandas: DataFrames, cleaning, grouping, aggregation
- Visualization: matplotlib basics

## Hands-on Path
- Quick Checks: choose stdlib modules; spot normalization issues
- Try This: read-delimited, parse JSON, fetch API
- Labs: weather ETL, API tracking, create a DB and query by 2 criteria

### Additional Topics (to align with requirements)
- Excel with `openpyxl`: read/write worksheets.
- XML with `xmltodict`: fetch, parse, and traverse.
- Pandas groupby/aggregation + matplotlib plotting.
- Sentinel processing: stop at a line containing `---`.
- NoSQL conceptual comparison: Redis (caching, sessions) vs MongoDB (documents).

### AI-comparison callout
For each lab, compare AI solutions on:
- Streaming vs reading entire file
- Avoiding repeated work inside loops
- Correct data type coercion and handling missing values

## Best Practices
- Stream/iterate large files (avoid reading all into memory)
- Validate and coerce types early; handle missing values
- Separate IO, transformation, and persistence layers
- Parameterize queries; avoid SQL injection
- Log steps and data assumptions for reproducibility

## Resources
- Python docs: csv, json, sqlite3, pathlib, requests
- Pandas user guide, matplotlib gallery, SQLAlchemy tutorial


