---
title: DataAnalyst Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DataAnalyst Agent (live demo)

Ask a Postgres database a business question in plain English and get the answer with the SQL, a
table and a chart. Bring your own free Gemini or Groq API key and a **read-only** database user.

Queries run read-only in two layers (a read-only session and a SQL guard that allows a single
SELECT). Your key and connection string live in your session's memory only.

Source, benchmark results and notes: https://github.com/dipdhru/DataAnalyst_AI_Agent
