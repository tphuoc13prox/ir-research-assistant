from pathlib import Path
from typing import Callable

import httpx


class Downloader:
    def __init__(self, *, timeout: float = 60.0, user_agent: str | None = None) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def download(
        self,
        url: str,
        output_dir: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        filename = filename.replace("/", "_")
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"
        output_path = output_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        import time
        retries = 5
        delay = 2.0
        for attempt in range(retries):
            try:
                with httpx.stream("GET", url, headers=self.headers, timeout=self.timeout, follow_redirects=True) as response:
                    response.raise_for_status()
                    with output_path.open("wb") as file:
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                return output_path
            except httpx.HTTPError as exc:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception:
                        pass
                if attempt == retries - 1:
                    raise exc
                
                # Check for 429 Too Many Requests
                backoff = delay * (2 ** attempt)
                err_msg = "Error downloading PDF"
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    try:
                        backoff = float(retry_after) if retry_after else 10.0
                    except ValueError:
                        backoff = 10.0
                    err_msg = f"Rate limited by arXiv (429). Retrying in {backoff:.1f}s (attempt {attempt + 1}/{retries})..."
                else:
                    err_msg = f"Network issue: {exc}. Retrying in {backoff:.1f}s (attempt {attempt + 1}/{retries})..."
                
                if progress_callback:
                    progress_callback(err_msg)
                
                time.sleep(backoff)
        return output_path
