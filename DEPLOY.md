# Deploying: GitHub

Put the project on GitHub

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
