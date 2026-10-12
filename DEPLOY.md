# Deploying: GitHub + HuggingFace Spaces

## Part A — Put the project on GitHub

1. On github.com click **New repository**. Name it e.g. `rag-document-qa`,
   set it **Public**, and leave "Add README / .gitignore / license"
   **unchecked** (they already exist here).
2. In the project folder:
   ```bash
   git init -b main
   git add .
   git commit -m "Initial commit: RAG document Q&A assistant"
   git remote add origin https://github.com/<your-username>/rag-document-qa.git
   git push -u origin main
   ```
   When asked for a password, use a **Personal Access Token**
   (GitHub → Settings → Developer settings → Tokens, scope `repo`), or run
   `gh auth login` first if you use the GitHub CLI.
3. Open the **Actions** tab: the `tests` workflow should run and go green
   (~3–5 min on first run).
4. Fill in the placeholders:
   - `README.md`: `<your-username>` in the clone command; add the Space URL once it exists.
   - `app/streamlit_app.py`: the sidebar line `[GitHub repo](#) · [Live demo](#)`.
   - `LICENSE`: check the name in the copyright line.
5. **Re-run `python src/evaluate.py` on your own machine** (it needs
   internet for the embedding model) and paste the numbers it prints
   into the README "Results" section, so every figure there is one you
   measured yourself.

## Part B — Deploy to HuggingFace Spaces (free CPU)

The YAML block at the top of `README.md` is the Space configuration
(Streamlit SDK, entry point `app/streamlit_app.py`, Python 3.11). Nothing
else needs renaming.

1. Create a free account at huggingface.co.
2. **Create the Space:** profile menu → *New Space* →
   - Name: `rag-document-qa`
   - SDK: **Streamlit**
   - Hardware: **CPU basic · 2 vCPU · 16 GB · free**
   - Visibility: Public
3. **Create an access token:** Settings → Access Tokens → *New token* →
   type **Write**. Copy it.
4. **Push the code** (from the project folder):
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/rag-document-qa
   git push space main
   ```
   Username = your HF username, password = the write token.
   If HF says the remote already contains commits (the auto-generated
   README), use `git push space main --force` — this is safe here
   because the Space is brand new and your README replaces theirs.
   (Alternative: *Files → Add file → Upload files* in the Space's web
   UI; drag in everything except `.git`.)
5. Open the Space's **App** tab. The first build installs the
   requirements (~5–10 min, torch is large). Watch **Logs** if it fails.
6. **First question is slow.** The embedding model (~80 MB) and
   `flan-t5-base` (~1 GB) download on first use, and the demo index is
   built automatically. After that, answers take a few seconds on CPU.
7. Add the Space URL to your GitHub README and the sidebar link.

### Optional: auto-sync GitHub → Space

Add your HF token as a GitHub repo secret named `HF_TOKEN`
(Settings → Secrets and variables → Actions), then create
`.github/workflows/sync-to-space.yml`:

```yaml
name: sync-to-space
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - run: git push --force https://<hf-username>:${{ secrets.HF_TOKEN }}@huggingface.co/spaces/<hf-username>/rag-document-qa main
```

### Free-tier things to know

- The Space **sleeps after ~48 h of inactivity** and reloads the models
  when someone wakes it (first request is slow again).
- Disk is **ephemeral**: URLs/PDFs a visitor loads live only in their
  session; nothing is saved.
- To speed up builds, add `--extra-index-url https://download.pytorch.org/whl/cpu`
  as the first line of `requirements.txt` (CPU-only torch).

### Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on a package | Open **Logs**, find the failing `pip` line; pin that package in `requirements.txt`. |
| "Runtime error" right after build | Logs tab → traceback. Most common: out of memory (use CPU basic 16 GB, not a smaller tier) or a model download timeout → *Restart this Space*. |
| PDF upload fails with a 403 | Known Streamlit-in-iframe quirk on some versions. Add `.streamlit/config.toml` with `[server]` / `enableXsrfProtection = false` and `enableCORS = false`, push again. |
| Space stuck on "Building" | *Settings → Factory reboot*. |
