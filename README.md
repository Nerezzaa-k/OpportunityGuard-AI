# OpportunityGuard AI — ML + LLM

A student prototype that combines a traditional ML text classifier with an optional Google Gemini LLM explanation layer.

## Run
1. Install Python 3.10+.
2. In this folder run:
   `python -m venv .venv`
   `.venv\\Scripts\\activate`
   `python -m pip install -r requirements.txt`
3. Configure an API key to enable LLM output:
   - Windows CMD (current terminal): `set GEMINI_API_KEY=your_api_key`
   - PowerShell: `$env:GEMINI_API_KEY="your_api_key"`
   - Or create `.streamlit/secrets.toml` with:
     `GEMINI_API_KEY = "your_api_key"`
     `GEMINI_MODEL = "gemini-2.5-flash"`
   Obtain a key from Google AI Studio and check current account availability, quotas and terms.
4. Run: `streamlit run app.py`

Without an API key, the ML and rule-based components still run; LLM generation is disabled. Never upload an API key or secrets file to GitHub.

## Limitations and responsible use
The included dataset is synthetic. Its evaluation metrics do not establish real-world scam-detection accuracy. The app cannot prove fraud or legitimacy; verify opportunities through official channels. Do not paste passwords, OTPs, UPI PINs or private information. The LLM can make mistakes.
