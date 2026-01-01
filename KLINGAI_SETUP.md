# KlingAI Custom Upload Setup Guide

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements.txt
```

This will install:
- `python-dotenv` - For environment variable management (required)
- `tryon` - KlingAI try-on API package (optional - fallback implementation included)

**Note:** The `tryon` package is not available on PyPI. If you have access to it, install it manually. Otherwise, the code includes a fallback implementation that uses direct API calls.

## API Key Configuration

1. Get your KlingAI API credentials:
   - Visit https://klingai.com
   - Sign up or log in to your account
   - Navigate to API Keys section
   - Generate a new API key and secret key

2. Create a `.env` file in the project root directory:
```bash
# Create .env file
touch .env
```

3. Add your API credentials to the `.env` file:
```
KLINGAI_API_KEY=your_api_key_here
KLINGAI_SECRET_KEY=your_secret_key_here
KLINGAI_BASE_URL=https://api.klingai.com
```

Replace `your_api_key_here` and `your_secret_key_here` with your actual credentials.

## Usage

1. Start the Flask application:
```bash
python new1.py
```

2. Access the application:
   - Open browser: `http://localhost:5000`
   - Sign in with any username/password (demo mode)
   - Select "Custom Uploads" from the dashboard

3. Upload images:
   - Upload a person image (photo of a person)
   - Upload a clothing image (photo of the clothing item)
   - Click "Generate Try-On"
   - Wait for the AI to process and generate the result

## Features

- Custom image uploads for person and clothing
- AI-powered virtual try-on using KlingAI
- Real-time preview of uploaded images
- High-quality generated results
- Multiple result images support

## Troubleshooting

### "KlingAI Not Available" Error
- Ensure you have set `KLINGAI_API_KEY` and `KLINGAI_SECRET_KEY` in your `.env` file
- Restart the Flask application after adding environment variables
- Verify your API credentials are correct

### Import Errors
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using Python 3.8 or higher

### API Errors
- Check your API key and secret key are valid
- Verify you have sufficient API credits/quota
- Check your internet connection

## File Structure

Uploaded files are stored in:
- `static/uploads/` - User uploaded images
- `static/outputs/` - Generated try-on results

