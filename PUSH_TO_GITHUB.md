# Pushing this project to GitHub

Target repo: **https://github.com/anushkagupta200615-jpg/CASM.git**

Run these from inside the `CUDA-Softmax` folder on your own computer (where you're
already logged in to GitHub). This uploads everything to the `CASM` repo.

```bash
git init
git add -A
git commit -m "CUDA Accelerated Softmax: naive -> shared -> warp-shuffle -> vectorized + fused attention"
git branch -M main
git remote add origin https://github.com/anushkagupta200615-jpg/CASM.git
git push -u origin main
```

If the repo already has commits and the push is rejected, pull first:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Authentication
When git asks for a password, use a **Personal Access Token**, not your account
password: GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained token → give it access to the `CASM` repo with **Contents: Read and
write** → paste it as the password.

If you have the GitHub CLI installed and authenticated (`gh auth login`), a single
command does it:

```bash
gh repo create --source=. --remote=origin --push   # if CASM doesn't exist yet
# or, if CASM already exists:
git init && git add -A && git commit -m "initial commit" && \
  git remote add origin https://github.com/anushkagupta200615-jpg/CASM.git && \
  git push -u origin main
```

## After pushing

- Open `run_on_colab.ipynb` in Google Colab (`Runtime → Change runtime type → T4 GPU`)
  and run it to fill in the real benchmark numbers.
- Paste those numbers into the README table and commit again:

```bash
git add README.md && git commit -m "Add real benchmark results" && git push
```

- Add repo **topics** on GitHub (`cuda`, `gpu`, `softmax`, `hpc`, `attention`) so it's
  discoverable, and link it in your LFX/GSoC application as evidence of GPU/systems ability.
