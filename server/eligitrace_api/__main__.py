"""`task app` -> uvicorn (no --reload: reloading would load the model twice)."""

import uvicorn

from server.eligitrace_api.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    print(f"EligiTrace starting on http://{s.host}:{s.port}  (API docs: /docs). Model loads in the background.")
    uvicorn.run("server.eligitrace_api.main:app", host=s.host, port=s.port, log_level="info")
