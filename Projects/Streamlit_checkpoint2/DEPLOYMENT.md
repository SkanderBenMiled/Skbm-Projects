# Deployment Guide for Streamlit Share

## Prerequisites
1. Install Git on your system
2. Create a GitHub account if you don't have one
3. Have a Streamlit Share account (free at share.streamlit.io)

## Steps to Deploy

### 1. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Financial inclusion Streamlit app"
```

### 2. Create GitHub Repository
1. Go to GitHub.com and create a new repository
2. Name it something like "financial-inclusion-streamlit"
3. Don't initialize with README (we already have files)

### 3. Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/financial-inclusion-streamlit.git
git branch -M main
git push -u origin main
```

### 4. Deploy on Streamlit Share
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your GitHub repository
5. Set the main file path to: `Financial_inclusion.py`
6. Click "Deploy"

### 5. Your App Will Be Live!
Streamlit Share will automatically:
- Install dependencies from requirements.txt
- Deploy your app to a public URL
- Update automatically when you push changes to GitHub

## Important Files for Deployment
- `Financial_inclusion.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `Financial_inclusion_dataset.csv` - Dataset file
- `.gitignore` - Git ignore rules

## Tips
- Make sure all file paths are relative (no absolute paths)
- Keep the dataset file small (< 100MB for GitHub)
- Test locally before deploying
- Check that all required packages are in requirements.txt

## Local Testing
To test locally before deployment:
```bash
pip install -r requirements.txt
streamlit run Financial_inclusion.py
```
