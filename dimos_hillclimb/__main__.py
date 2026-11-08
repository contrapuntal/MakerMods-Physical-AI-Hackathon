"""Run the Hill Climb relay through DimOS.

    # skills + MCP server only (no LLM key needed); then drive it from another terminal
    repos/dimos/.venv/bin/python -m dimos_hillclimb serve --seed 1
    repos/dimos/.venv/bin/python -m dimos_hillclimb drive            # scripted planner over MCP
    repos/dimos/.venv/bin/dimos mcp call look                        # or call skills by hand

    # skills + MCP server + DimOS's LLM agent, which plans the match itself
    ANTHROPIC_API_KEY=... repos/dimos/.venv/bin/python -m dimos_hillclimb agent --model anthropic:claude-opus-5-5
    repos/dimos/.venv/bin/dimos agent-send "Go. Our colour is orange."

Run from the workspace root (the folder holding dimos_hillclimb/ and sim/).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dimos_hillclimb", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "agent"):
        p = sub.add_parser(name)
        p.add_argument("--backend", default="sim", choices=("sim", "station"))
        p.add_argument("--color", default="orange", help="our ball colour from the coin flip")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--time-limit", type=float, default=300.0)
        p.add_argument("--station-url", default="http://127.0.0.1:8765",
                       help="station backend: the station server (python -m harness serve on the station)")
        p.add_argument("--dashboard-host", default="127.0.0.1", help="0.0.0.0 to open the dashboard from the LAN")
        p.add_argument("--dashboard-port", type=int, default=8770, help="0 for no dashboard")
        if name == "agent":
            p.add_argument("--model", default="anthropic:claude-opus-5-5",
                           help="any LangChain model id, e.g. anthropic:claude-opus-5-5 or gpt-5.6-luna")
    d = sub.add_parser("drive", help="scripted planner: plays the match through the MCP tools")
    d.add_argument("--url", default="http://127.0.0.1:9990/mcp")
    d.add_argument("--color", default="")
    d.add_argument("--seed", type=int, default=-1)
    d.add_argument("--station", action="store_true", help="the real station's tower plan (server is --backend station)")
    args = ap.parse_args(argv)

    if args.cmd == "drive":
        from dimos_hillclimb.drive import drive
        sys.exit(0 if drive(args.url, args.color, args.seed, args.station) else 1)

    _require_free_ports([("the MCP server", 9990)] + ([("the dashboard", args.dashboard_port)]
                                                      if args.dashboard_port else []))
    # DimOS deploys modules into worker processes; they must import dimos_hillclimb too.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    from dimos_hillclimb import blueprints

    if args.cmd == "serve":
        bp = blueprints.hillclimb_mcp(args.backend, args.color, args.seed, args.time_limit, args.station_url,
                                      args.dashboard_host, args.dashboard_port)
    else:
        bp = blueprints.hillclimb_agent(args.model, args.backend, args.color, args.seed, args.time_limit,
                                        args.station_url, args.dashboard_host, args.dashboard_port)
    from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
    from dimos.core.coordination.module_coordinator import ModuleCoordinator

    # serve_coordinator_rpc off: don't claim the bus-wide Coordinator name, so this runs beside another DimOS run
    # (e.g. the dual-metal planner coordinator). `dimos status/stop` won't see it; Ctrl-C stops it.
    parsed = BlueprintConfigParser(bp).parse(
        environ=dict(os.environ), overrides={"g": {"viewer": "none", "serve_coordinator_rpc": False}})
    ModuleCoordinator.build(bp, parsed).loop()


def _require_free_ports(ports: list[tuple[str, int]]) -> None:
    """Exits with a plain message when another run holds our ports. Two hillclimb runs also share module names on
    DimOS's bus, so the second one's start calls would reach the first one's modules: one run at a time."""
    import socket

    for what, port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                sys.exit(f"port {port} ({what}) is in use: another dimos_hillclimb run (serve or agent) is still up. "
                         "Stop it with Ctrl-C first; only one run at a time.")


if __name__ == "__main__":
    main()
