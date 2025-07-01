# Google Custom Search Engine (CSE) Setup Guide

## 🔧 Step 1: Create Google CSE

1. **Go to**: [https://programmablesearchengine.google.com](https://programmablesearchengine.google.com)

2. **Create a new search engine**:

   - Click "Add" button
   - Enter sites to search (recommended financial sites):
     - `reuters.com`
     - `bloomberg.com`
     - `marketwatch.com`
     - `cnbc.com`
     - `yahoo.com/finance`
     - `investing.com`
     - `forexfactory.com`
   - Give it a name like "Financial News Search"
   - Click "Create"

3. **Enable "Search the entire web"**:

   - Go to **Control Panel → Sites to Search → Advanced**
   - Toggle "Search the entire web" to ON
   - This allows broader search capabilities while prioritizing your specified sites

4. **Get your Search Engine ID**:
   - In the Control Panel, go to "Overview"
   - Copy the **Search engine ID (cx)** - it looks like: `012345678901234567890:abcdefghijk`

## 🔑 Step 2: Get Google API Key

1. **Go to**: [https://console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)

2. **Create or select a project**

3. **Enable the Custom Search API**:

   - Go to [API Library](https://console.cloud.google.com/apis/library)
   - Search for "Custom Search API"
   - Click on it and click "Enable"

4. **Create API Key**:
   - Go back to Credentials
   - Click "Create Credentials" → "API Key"
   - Copy the API key - it looks like: `AIzaSyABC123DEF456GHI789JKL012MNO345PQR`

## 🔧 Step 3: Configure Environment Variables

Add these to your environment variables (`.env` file or system environment):

```bash
# Google Custom Search Engine
GOOGLE_CSE_API_KEY=AIzaSyABC123DEF456GHI789JKL012MNO345PQR
GOOGLE_CSE_ID=012345678901234567890:abcdefghijk
```

## 🚀 Step 4: Test the Integration

1. **Start your trading bot server**:

   ```bash
   cd python_tradingbot
   python master_server.py
   ```

2. **Test in the AI Chat**:
   - Ask questions like:
     - "What's the latest EUR/USD news?"
     - "What's happening in the markets today?"
     - "Tell me about recent Fed policy decisions"
3. **Verify web search is working**:
   - Look for sources appearing in the chat responses
   - Check the server logs for "Performing web search" messages

## 📊 Usage Quotas

- **Free Tier**: 100 search queries per day
- **Paid Tier**: $5 per 1,000 queries (up to 10,000 per day)

## 🔍 How It Works

The system automatically detects when web search is needed based on:

- **Keywords**: "latest", "current", "today", "news", "breaking", etc.
- **Financial terms**: "price", "market", "trading", "forex", "crypto", etc.
- **Question patterns**: "What's happening...", "What's the latest...", etc.
- **Instrument mentions**: EUR/USD, Bitcoin, Gold, etc.

When triggered, it:

1. Searches financial news sites
2. Formats results for AI context
3. AI provides analysis with current information
4. Sources are displayed in the chat interface

## 🛡️ Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Consider using Google Cloud Secret Manager for production
- Monitor API usage to avoid unexpected charges

## 🔧 Troubleshooting

**No search results appearing?**

1. Check environment variables are set correctly
2. Verify API key has Custom Search API enabled
3. Check server logs for error messages
4. Test API key with a simple curl request:

```bash
curl "https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_CSE_ID&q=test"
```

**Getting quota exceeded errors?**

- Check your Google Cloud Console for usage limits
- Consider upgrading to paid tier if needed
- Implement caching to reduce API calls

## 📈 Advanced Configuration

You can customize the search behavior by modifying `services/google_search_service.py`:

- **Add more financial sites** to the search query
- **Adjust search parameters** (date restrictions, result count)
- **Modify keyword detection** for different query types
- **Add caching** to reduce API calls for repeated queries
