from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the first training/inference web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", default=7860, type=int, help="bind port")
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the local Web UI in the default browser",
    )
    args = parser.parse_args()
    try:
        from launcher.webui import launch
    except ModuleNotFoundError as exc:
        if exc.name == "gradio":
            raise SystemExit("Missing dependency: gradio. Run `python -m pip install -r requirements.web.txt` first.")
        raise
    launch(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
