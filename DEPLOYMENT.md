# LexiLLM Deployment Guide

This document provides step-by-step instructions for deploying LexiLLM to GitHub and Streamlit Cloud.

## GitHub Deployment

1. Create a new GitHub repository:
   - Go to [GitHub](https://github.com/) and sign in
   - Click on the "+" icon in the top right and select "New repository"
   - Name your repository (e.g., "LexiLLM")
   - Choose public or private visibility
   - Click "Create repository"

2. Initialize Git in your local project (if not already done):
   ```bash
   cd /path/to/LexiLLM
   git init
   ```

3. Set the remote repository:
   ```bash
   git remote add origin https://github.com/yourusername/LexiLLM.git
   ```

4. Stage and commit your files:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```

5. Push to GitHub:
   ```bash
   git push -u origin main
   ```

## Streamlit Cloud Deployment

1. Sign up or log in to [Streamlit Cloud](https://streamlit.io/cloud).

2. Connect to your GitHub repository:
   - Click "New app"
   - Select your GitHub repository
   - In the "Main file path" field, enter `src/ui_streamlit_enhanced.py`

3. Configure advanced settings:
   - Set Python version to 3.9 or higher
   - Add any required packages if they're not in requirements.txt

4. Set up secrets:
   - In the Streamlit Cloud dashboard, go to your app's settings
   - Click "Secrets" in the sidebar
   - Add your secrets in TOML format:
     ```toml
     OPENAI_API_KEY = "your_api_key_here"
     ```

5. Deploy:
   - Click "Deploy"
   - Wait for the build and deployment process to complete

6. Access your app:
   - Once deployment is complete, you'll get a URL to access your app
   - Share this URL with others to let them use LexiLLM

## Local Development with Streamlit

1. Create a `.streamlit/secrets.toml` file (already done in your project):
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

2. Run the Streamlit app locally:
   ```bash
   streamlit run src/ui_streamlit_enhanced.py
   ```

## Updating the Deployed App

1. Make and test your changes locally.

2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update message"
   git push
   ```

3. Streamlit Cloud will automatically detect the changes and rebuild the app.

## Troubleshooting

1. **API Key Issues**:
   - Ensure your API key is correctly set in Streamlit Cloud secrets
   - Check the logs in Streamlit Cloud for authentication errors

2. **Dependency Issues**:
   - Make sure all required packages are listed in requirements.txt
   - Check version compatibility if you encounter errors

3. **File Path Issues**:
   - Ensure all file paths are relative and not hardcoded to local directories
   - Use `os.path.join` for cross-platform compatibility

4. **Memory Errors**:
   - Be mindful of the memory limitations in Streamlit Cloud
   - Optimize your code to reduce memory usage if needed
