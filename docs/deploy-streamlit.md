# Hosting the public demo on Streamlit Community Cloud

The demo is the Streamlit UI in **demo mode** (`ANALYST_DEMO=1`). Visitors bring their own free
Gemini or Groq key and a **read-only** Postgres connection string. The server holds no model key
and no database of its own.

Live at https://dipdhru-dataanalyst-ai-agent-uiapp-e4j7an.streamlit.app

## What demo mode does

- **One workspace per browser session** (`src/analyst/demo.py`): its own temp state folder,
  connections, catalog, memory and chat history, deleted when the session ends. Nothing is
  shared between visitors.
- **Host vetting** (`src/analyst/netguard.py`): the server only connects to public addresses over
  TLS. Loopback, private ranges, link-local (cloud metadata) and carrier-grade NAT are refused,
  URL options that could redirect the connection are refused, and the connection is pinned to the
  address that was checked.
- **No server credentials reachable**: sources from `config.yaml` other than local SQLite sample
  files are not offered, and the model client only ever uses the key the visitor typed.
- Queries still go through `Connector.run_query` (read-only session plus `sql_guard`).

## Deploying

1. Go to share.streamlit.io, sign in with GitHub and choose **Create app**.
2. Repository `dipdhru/DataAnalyst_AI_Agent`, branch `main`, main file `ui/app.py`.
3. Under **Advanced settings** choose Python 3.12 and add this secret:

   ```toml
   ANALYST_DEMO = "1"
   ```

4. Deploy. The app redeploys on every push to `main`.
5. Under Settings > Sharing, make sure the app is **public**.

Dependencies come from `uv.lock`. Streamlit advises one dependency file per app, so don't add a
`requirements.txt`.

## Known limits

- **Sleeps** after about 12 hours without visitors; the next visit wakes it.
- **Credentials pass through the server.** The sidebar says so. Advise a read-only user on a copy
  of the data.
- **Query results go to the model provider the visitor chose.**
- **No rate limit per visitor.** Their own key pays for model calls, but a visitor can still make
  the server run queries against their own database. Auth and an audit log are phase 6.
- **No sample database ships with the app.** Olist is 230 MB and under a non-commercial licence,
  so a visitor without a database of their own has nothing to try.
