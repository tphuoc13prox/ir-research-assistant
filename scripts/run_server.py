from pathlib import Path
import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _open_browser(host: str, port: int) -> None:
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{port}"
    
    # Cho den khi server phan hoi (toi da 10 giay)
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            result = sock.connect_ex((browser_host, port))
            if result == 0:
                time.sleep(0.3)
                webbrowser.open(url)
                return
        time.sleep(0.2)
    
    # Fallback
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local IR Research Assistant web app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    port = _find_available_port(args.host, args.port)

    if port != args.port:
        print(f"Port {args.port} is unavailable. Using http://{args.host}:{port}")
    else:
        print(f"Running at http://{args.host}:{port}")

    # Mo browser tu dong trong thread khac de khong block uvicorn
    threading.Thread(
        target=_open_browser,
        args=(args.host, port),
        daemon=True,
    ).start()

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=port,
        reload=not args.no_reload,
        reload_dirs=[str(PROJECT_ROOT)],
    )


def _find_available_port(
    host: str,
    preferred_port: int,
    attempts: int = 20,
) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port

    raise RuntimeError(
        f"No available port found from "
        f"{preferred_port} to {preferred_port + attempts - 1}"
    )


if __name__ == "__main__":
    main()
