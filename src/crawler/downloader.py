from pathlib import Path

import httpx


class Downloader:
    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def download(self, url: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        filename = filename.replace("/", "_")
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"
        output_path = output_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        with httpx.stream("GET", url, timeout=self.timeout, follow_redirects=True) as response:
            response.raise_for_status()
            with output_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        return output_path
