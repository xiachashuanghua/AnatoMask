from __future__ import annotations

import argparse
import inspect
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_TITLE = "AnatoMask"
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAVICON_PATH = PROJECT_ROOT / "logo_no_cha.png"
CLIENT_ICON_PATH = PROJECT_ROOT / "webui_runs" / "client_icon.ico"


def _connect_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return bind_host


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _ephemeral_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _resolve_port(host: str, preferred_port: int) -> tuple[int, bool]:
    if preferred_port > 0 and _is_port_available(host, preferred_port):
        return preferred_port, False

    fallback_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    fallback_port = _ephemeral_port(fallback_host)
    return fallback_port, preferred_port > 0


def _wait_for_server(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    if last_error is None:
        raise TimeoutError(f"Timed out while waiting for {url}")
    raise RuntimeError(f"Desktop client could not reach {url}: {last_error}")


def _resolve_window_icon() -> str | None:
    if not FAVICON_PATH.exists():
        return None

    if os.name != "nt":
        return str(FAVICON_PATH)

    try:
        from PIL import Image

        CLIENT_ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        if (
            not CLIENT_ICON_PATH.exists()
            or CLIENT_ICON_PATH.stat().st_mtime < FAVICON_PATH.stat().st_mtime
        ):
            with Image.open(FAVICON_PATH) as image:
                image.save(
                    CLIENT_ICON_PATH,
                    format="ICO",
                    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
                )
        return str(CLIENT_ICON_PATH)
    except Exception as exc:
        print(f"[AnatoMask] Failed to prepare the desktop icon from {FAVICON_PATH}: {exc}")
        return None


def _server_thread(bind_host: str, port: int, result: dict, ready: threading.Event) -> None:
    try:
        from launcher.webui import build_app

        demo = build_app()
        launch_kwargs = {
            "server_name": bind_host,
            "server_port": port,
            "prevent_thread_lock": True,
            "inbrowser": False,
            "show_api": False,
        }
        if FAVICON_PATH.exists():
            launch_kwargs["favicon_path"] = str(FAVICON_PATH)
        try:
            signature = inspect.signature(demo.launch)
            launch_kwargs = {
                name: value
                for name, value in launch_kwargs.items()
                if name in signature.parameters
            }
        except (TypeError, ValueError):
            pass
        demo.launch(**launch_kwargs)
        result["demo"] = demo
    except Exception as exc:
        result["error"] = exc
    finally:
        ready.set()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the AnatoMask desktop client.")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST, help="local bind host for the embedded web server")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="local bind port for the embedded web server")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="window title")
    parser.add_argument("--width", default=1440, type=int, help="window width")
    parser.add_argument("--height", default=960, type=int, help="window height")
    parser.add_argument("--min-width", default=1100, type=int, help="minimum window width")
    parser.add_argument("--min-height", default=780, type=int, help="minimum window height")
    parser.add_argument("--debug", action="store_true", help="enable pywebview debug mode")
    args = parser.parse_args(argv)

    try:
        import webview
    except ModuleNotFoundError as exc:
        if exc.name == "webview":
            raise SystemExit(
                "Missing dependency: pywebview. Run the client startup script or install client/requirements.txt first."
            ) from exc
        raise

    resolved_port, used_fallback = _resolve_port(args.host, args.port)
    if used_fallback:
        print(f"[AnatoMask] Port {args.port} is unavailable. Falling back to {resolved_port}.")

    connect_host = _connect_host(args.host)
    url = f"http://{connect_host}:{resolved_port}"
    result: dict[str, object] = {}
    ready = threading.Event()
    thread = threading.Thread(
        target=_server_thread,
        args=(args.host, resolved_port, result, ready),
        daemon=True,
        name="anatomask-gradio",
    )
    thread.start()
    if not ready.wait(timeout=30.0):
        raise SystemExit("Timed out while starting the embedded AnatoMask web server.")
    if "error" in result:
        raise SystemExit(f"Failed to start the embedded AnatoMask web server: {result['error']}")

    _wait_for_server(url)

    webview.create_window(
        args.title,
        url,
        width=args.width,
        height=args.height,
        min_size=(args.min_width, args.min_height),
        confirm_close=True,
    )
    try:
        webview_start_kwargs = {"debug": args.debug}
        window_icon = _resolve_window_icon()
        if window_icon:
            webview_start_kwargs["icon"] = window_icon
        try:
            start_signature = inspect.signature(webview.start)
            webview_start_kwargs = {
                name: value
                for name, value in webview_start_kwargs.items()
                if name in start_signature.parameters
            }
        except (TypeError, ValueError):
            pass
        webview.start(**webview_start_kwargs)
    finally:
        demo = result.get("demo")
        if demo is not None:
            demo.close()
