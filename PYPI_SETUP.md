# PyPI Publishing Setup Guide

## ✅ API Token Stored Securely

Your PyPI API token has been saved to `.env.pypi` (which is gitignored).

## 🔐 Add Token to GitHub Secrets

To use this token in GitHub Actions for automated publishing:

### Step 1: Copy the Token
Open `.env.pypi` and copy the token value (starts with `pypi-...`)

### Step 2: Add to GitHub Repository Secrets

1. Go to your repository on GitHub:
   https://github.com/moshesham/AI-Omniscient-Architect

2. Navigate to: **Settings** → **Secrets and variables** → **Actions**

3. Click **"New repository secret"**

4. Add the secret:
   - **Name**: `PYPI_API_TOKEN`
   - **Value**: (paste the token from `.env.pypi`)

5. Click **"Add secret"**

### Step 3: (Optional) Add TestPyPI Token

If you have a separate TestPyPI token:

1. Create another secret:
   - **Name**: `TESTPYPI_API_TOKEN`
   - **Value**: (your TestPyPI token)

## 📦 Manual Publishing (Alternative)

If you prefer to publish manually without GitHub Actions:

```bash
# Build all packages
python -m build

# Install twine if needed
pip install twine

# Upload to TestPyPI (for testing)
python -m twine upload --repository testpypi dist/*
# When prompted for token, use the value from .env.pypi

# Upload to PyPI (production)
python -m twine upload dist/*
# When prompted for token, use the value from .env.pypi
```

## 🔄 Update GitHub Workflow

The workflow has been updated to use the environment-based secrets. You can now:

1. **Tag a release** to trigger automatic publishing:
   ```bash
   # For meta-package
   git tag v0.2.0
   git push origin v0.2.0
   
   # For individual packages
   git tag omniscient-core-v0.1.0
   git push origin omniscient-core-v0.1.0
   ```

2. **Manual trigger** via GitHub Actions tab:
   - Go to Actions → "Publish to PyPI" workflow
   - Click "Run workflow"
   - Select package and options

## ⚠️ Security Notes

1. **Never commit** `.env.pypi` to version control
2. **Never share** your API token publicly
3. **Rotate tokens** periodically for security
4. The token in `.env.pypi` is for local reference only
5. GitHub Actions should use the **GitHub Secrets** version

## 🎯 Next Steps

1. ✅ Token stored securely in `.env.pypi`
2. ⏳ Add `PYPI_API_TOKEN` to GitHub Secrets (see Step 2 above)
3. ⏳ Test publishing with a tag or manual workflow trigger
4. ⏳ Configure trusted publishers (optional but recommended)

Once the GitHub secret is added, your workflow will be able to publish packages automatically!
