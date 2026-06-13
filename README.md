# Arabic Text-to-SQL using RAG, Gemini, and Silma Matrioshka

## Overview

This project implements an Arabic Natural Language to SQL Translation system using a Retrieval-Augmented Generation (RAG) architecture. The goal is to enable Arabic-speaking users to interact with databases using natural language instead of writing SQL queries manually.

The system combines semantic retrieval with Large Language Models (LLMs) to generate accurate SQL statements from Arabic questions. It utilizes Silma Matrioshka Embeddings for semantic search, Google Gemini for SQL generation, and FastAPI for serving the application.

---

## Features

* Convert Arabic natural language questions into SQL queries.
* Retrieval-Augmented Generation (RAG) pipeline.
* Semantic search using vector embeddings.
* Google Gemini integration for SQL generation.
* FastAPI REST API endpoints.
* Arabic language support.
* Example retrieval for context-aware SQL generation.

---

## System Architecture

```text
User Question (Arabic)
          │
          ▼
  Silma Matrioshka
     Embeddings
          │
          ▼
   Vector Database
 (Semantic Retrieval)
          │
          ▼
 Retrieve Top-K Similar
 Question-SQL Examples
          │
          ▼
 Prompt Construction
          │
          ▼
    Google Gemini
          │
          ▼
 Generated SQL Query
          │
          ▼
      FastAPI API
```

---

## Dataset

The system is built using the Arabic version of the Spider Dataset, a cross-domain semantic parsing benchmark.

Dataset format:

```json
{
  "question": "ما هي أسماء الطالب الذين حضروا جميع الصفوف؟",
  "query": "SELECT ...",
  "db_id": "university"
}
```

### Dataset Fields

| Field    | Description                      |
| -------- | -------------------------------- |
| question | Arabic natural language question |
| query    | Corresponding SQL query          |
| db_id    | Target database identifier       |

---

## Methodology

### 1. Text Embedding

Each Arabic question is converted into a vector representation using:

* Silma Matrioshka Embedding Model

These embeddings capture semantic meaning and enable similarity-based retrieval.

### 2. Vector Database

Question-SQL pairs are stored as vectorized chunks.

The vector database allows:

* Semantic similarity search
* Fast retrieval of relevant examples
* Context-aware SQL generation

### 3. Retrieval-Augmented Generation (RAG)

The RAG workflow consists of:

1. User submits an Arabic question.
2. The question is embedded.
3. Similar examples are retrieved from the vector database.
4. Retrieved examples are injected into the prompt.
5. Google Gemini generates the SQL query.
6. The generated SQL query is returned to the user.

### 4. Prompt Engineering

Carefully designed prompts guide Gemini to:

* Understand Arabic user intent.
* Utilize retrieved examples.
* Generate syntactically correct SQL.
* Follow database schema constraints.

---

## API Endpoints

### Translate Arabic Question

```http
POST /translate
```

Request:

```json
{
  "question": "كم عدد الطلاب الذين سجلوا في مادة الرياضيات؟"
}
```

Response:

```json
{
  "sql": "SELECT COUNT(*) FROM enrollment WHERE course='رياضيات';"
}
```

---

### Retrieve Examples

```http
GET /examples
```

Returns sample Question-SQL pairs for testing and debugging.

---

## Technologies Used

* Python
* FastAPI
* Google Gemini
* Retrieval-Augmented Generation (RAG)
* Silma Matrioshka Embeddings
* Vector Database
* JSONL Dataset
* SQL
* Natural Language Processing (NLP)

---

## Example

### Input

```text
كم عدد الطلاب الذين سجلوا في مادة الرياضيات؟
```

### Generated SQL

```sql
SELECT COUNT(*) 
FROM enrollment
WHERE course = 'رياضيات';
```

---

## Challenges

* Handling ambiguous Arabic expressions.
* Aligning generated SQL with target database schemas.
* Improving retrieval quality for unseen questions.
* Ensuring SQL correctness across different domains.

---

## Future Improvements

* Support multiple SQL dialects.
* Fine-tune Arabic embedding models.
* Add schema-aware retrieval.
* Integrate execution feedback for query validation.
* Develop a web-based user interface.

---

## Conclusion

This project demonstrates the effectiveness of combining Retrieval-Augmented Generation with Large Language Models for Arabic Text-to-SQL tasks. By leveraging semantic retrieval and contextual prompting, the system generates accurate and meaningful SQL queries from natural language Arabic inputs, making database interaction more accessible for Arabic-speaking users.
