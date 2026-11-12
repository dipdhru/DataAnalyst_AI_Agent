# Deploying the public demo on Hugging Face

The demo is the Streamlit UI in **demo mode** (`ANALYST_DEMO=1`). Visitors bring their own free
Gemini or Groq key and a **read-only** Postgres connection string. The server holds no model key
and no database of its own.

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

## Steps

1. Create a Space at huggingface.co/new-space: SDK **Docker**, any hardware (the free CPU tier is
   enough).
2. Build the Space files:

   ```bash
   uv run python scripts/build_space.py build/space
   ```

3. Push them to the Space (needs your Hugging Face login or a write token):

   ```bash
   cd build/space
   git init && git add . && git commit -m "Demo"
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push --force space HEAD:main
   ```

4. Wait for the build, open the Space, paste a key and a connection string.

Re-run steps 2-3 to update; `build/` is git-ignored.

## Known limits

- **Untested on Hugging Face itself.** The tests cover demo mode, and the packaged files were
  smoke-run locally, but the Docker image has not been built (no Docker on the dev machine).
- **Credentials pass through the server.** Say so to visitors (the sidebar does). Advise a
  read-only user on a copy of the data.
- **Query results go to the model provider the visitor chose.**
- **No rate limit per visitor.** Their own key pays for model calls, but a visitor can still make
  the server run queries against their own database. Auth and an audit log are phase 6.
- **Free Spaces sleep** after inactivity and the first visit wakes them (about a minute).
- **No sample database ships with the Space.** Olist is 230 MB and under a non-commercial
  licence. A visitor without a database of their own has nothing to try; consider a small
  hosted demo database if you want a zero-setup path.
