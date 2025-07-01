### Migrating your Electron + Python desktop app from OANDA to IG — a **battle-tested implementation plan**

---

#### 1 — Prep work (one afternoon)

1. **Open an IG demo account** → _My IG ▸ Settings ▸ API keys_ → generate an **API key**.
2. **Register an OAuth client** (same portal)

   - Add a **Redirect URI** that works on desktop — e.g.

     - `http://127.0.0.1:8787/ig/oauth-callback` (loop-back), **or**
     - a custom scheme `myapp://ig/callback` (Electron can register it).

3. Note the **client_id** and **client_secret**.

---

#### 2 — High-level architecture

```text
┌─────────── Electron (JS) ───────────┐
│                                      │
│  ① “Connect IG” button               │
│        │ shell.openExternal(authURL) │
│        ▼                             │
│  Browser shows IG login/consent      │
│        │                             │
│  ② IG redirects to localhost/custom  │
└────────┬─────────────────────────────┘
         │  (code, state) JSON POST
         ▼
┌───────────── Python backend ─────────────┐
│  ③ POST /ig/exchange-code               │
│      → POST /oauth2/access_token        │
│         (client_id, client_secret...)   │
│      ← access_token + refresh_token     │
│  ④ GET /session?fetchSessionTokens=true │
│      ← CST + X-SECURITY-TOKEN           │
│  ⑤ Store tokens in OS-keychain/KMS      │
│  ⑥ WebSocket ↔ Lightstreamer (prices)   │
│  ⑦ REST calls (orders, history, etc.)   │
└──────────────────────────────────────────┘
```

Sources: IG OAuth + `/session v3` tokens ([labs.ig.com][1]), streaming token upgrade flow ([labs.ig.com][2]), IG OAuth sample repo ([github.com][3])

---

#### 3 — Concrete code skeletons

**Electron (main process)**

```javascript
// auth.js
const { shell, ipcMain, app } = require('electron');
const axios = require('axios');
const PORT = 8787;

// 1. Ask backend for the auth URL
ipcMain.handle('ig-auth-url', async () => {
  const { data } = await axios.get('http://127.0.0.1:5000/ig/auth-url');
  shell.openExternal(data.url); // launches user’s default browser
});

// 2. Local HTTP listener to catch the redirect
const http = require('http');
http
  .createServer(async (req, res) => {
    if (req.url.startsWith('/ig/oauth-callback')) {
      const code = new URLSearchParams(req.url.split('?')[1]).get('code');
      await axios.post('http://127.0.0.1:5000/ig/exchange-code', { code });
      res.end('✅ IG account linked — you can close this tab.');
    }
  })
  .listen(PORT);
```

**Python backend (FastAPI snippet)**

```python
# ig_service.py
import httpx, time, os, json
API_KEY       = os.getenv("IG_API_KEY")
CLIENT_ID     = os.getenv("IG_CLIENT_ID")
CLIENT_SECRET = os.getenv("IG_CLIENT_SECRET")
REDIRECT_URI  = "http://127.0.0.1:8787/ig/oauth-callback"
AUTH_BASE     = "https://demo-api.ig.com"

def auth_url(state: str):
    return (f"{AUTH_BASE}/oauth2/authorize?"
            f"response_type=code&client_id={CLIENT_ID}"
            f"&redirect_uri={REDIRECT_URI}&state={state}&scope=profile")

async def exchange_code(code: str):
    async with httpx.AsyncClient() as client:
        token_r = await client.post(f"{AUTH_BASE}/oauth2/access_token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        })
        tok = token_r.json()
        # persist {access, refresh, expiry}
        save_tokens(tok)
        # Lightstreamer tokens
        sess = await client.get(f"{AUTH_BASE}/gateway/deal/session",
                                headers={"Authorization": f"Bearer {tok['access_token']}"},
                                params={"fetchSessionTokens": "true"})
        cst  = sess.headers["CST"]
        xst  = sess.headers["X-SECURITY-TOKEN"]
        save_stream_tokens(cst, xst, tok['expires_in'])
```

_(Full repo example linked above)_

---

#### 4 — Bridging your existing OANDA logic

| OANDA concept                             | IG equivalent                                  | Migration tip                                                  |
| ----------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| Instrument `EUR_USD`                      | **EPIC** like `CS.D.EURUSD.CFD.IP`             | GET `/markets?searchTerm=EUR/USD` to resolve once, then cache. |
| Order `units`                             | `size` (lots)                                  | `size = abs(units)`, `direction = BUY/SELL`                    |
| REST prices `/v3/instruments/.../candles` | `/prices?epic=...&resolution=MINUTE_5&max=...` | Same JSON→pandas parsing.                                      |
| Streaming `/pricing/stream`               | **Lightstreamer** item `MARKET:{EPIC}`         | Use `lightstreamer-client` (JS) or `lightstreamer-python`.     |

Add an **adapter layer**:

```python
class BrokerAdapter(ABC):
    def quote(self, symbol): ...
    def place_market(self, symbol, qty, side): ...
    ...

class IGBroker(BrokerAdapter): ...
class OANDABroker(BrokerAdapter): ...
```

Switching the backend is then a one-liner.

---

#### 5 — Token storage & refresh

- Store `access_token`, `refresh_token`, `CST`, `XST` in an encrypted vault (OS keyring or AWS KMS).
- Refresh access tokens every \~50 s (`expires_in` ≈ 60 s) — IG returns a **new refresh token** each time.
- On refresh success, **also regenerate CST/XST** (`GET /session?fetchSessionTokens=true`).
- Automatically reconnect Lightstreamer on token roll-over.

---

#### 6 — UX tweaks for a desktop app

| Step                 | UI element                                                 | Notes                                                      |
| -------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| “Connect IG Account” | Button opens default browser; shows IG login/consent page. | Use `shell.openExternal` so 2FA works in the real browser. |
| Redirect caught      | Mini web-server (loop-back) or custom URI.                 | Loop-back avoids Windows URI-handler quirks.               |
| Success toast        | “IG linked! Pulling your account…”                         | Good moment to run `/accounts` to verify.                  |
| Re-authorise         | “Connection expired – click to re-connect”                 | Trigger full OAuth again if refresh fails.                 |

---

#### 7 — Rough timeline

| Task                                    | Est. time                                  |
| --------------------------------------- | ------------------------------------------ |
| IG developer setup + keys               | 1 h                                        |
| Python wrapper (login, refresh, orders) | 1–2 days                                   |
| Lightstreamer integration               | 0.5 day (polling fallback works meanwhile) |
| Electron OAuth plumbing                 | 0.5 day                                    |
| Instrument/order mapping layer          | 1 day                                      |
| QA with demo & live accounts            | 1 day                                      |

**Total:** \~1 work-week to a functional MVP.

---

### Bottom line

- **IG ticks your boxes** (global CFD broker, OAuth, API for EU/Japan).
- Technical lift is very manageable inside your current Electron + Python stack.
- Start with REST-only polling; add Lightstreamer once the core trades flow.

Ping me if you’d like the full FastAPI project template or a Lightstreamer example class.

[1]: https://labs.ig.com/rest-trading-api-guide.html 'REST trading API guide | IG Labs'
[2]: https://labs.ig.com/streaming-api-guide.html?utm_source=chatgpt.com 'Streaming API Guide - IG Labs'
[3]: https://github.com/IG-Group/ig-oauth-api-example 'GitHub - IG-Group/ig-oauth-api-example'

├── master_server.py # Main server (ALL 5 features)
├── master_trading_bot.py # Master AI Bot logic
├── oanda_trader.py # Broker integration
├── requirements.txt # Dependencies
├── NEXT_STEPS.md # IG migration plan
├── routes/
│ └── ai_analysis_routes.py # AI Analysis feature
├── services/
│ ├── simple_ai_service.py # AI Chat & Analysis
│ ├── custom_bot_service.py # Trading Bots execution
│ ├── supabase_service.py # Database & templates
│ ├── market_data_service.py # OANDA price feeds
│ ├── google_search_service.py # AI Chat web search
│ └── cache_service.py # Performance caching
└── config/
├── **init**.py # API keys & config
├── api_config.py # Validation
└── oanda_config.py # OANDA credentials
