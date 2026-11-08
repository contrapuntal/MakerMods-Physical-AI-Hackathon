"""A scripted planner that plays the match through the DimOS MCP tools, exactly as the LLM agent would call them.

It needs no model key, so it is the end-to-end test of the DimOS path (MCP server -> skills -> station -> referee),
and the fallback planner if the LLM is too slow or unreliable. It recovers from failures the simple way: look at the
table, redo whatever the tower is missing, retry a ball, give up on a piece after a few tries.
"""

from __future__ import annotations

import json
import time

from dimos.agents.mcp.mcp_adapter import McpAdapter

TRIES = 3


class Driver:
    def __init__(self, url: str) -> None:
        self.mcp = McpAdapter(url, timeout=120)

    def call(self, tool: str, **args) -> dict:
        t = time.monotonic()
        text = self.mcp.call_tool_text(tool, args)
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            out = {"message": text}
        msg = out.get("message", "")
        if tool != "look":
            print(f"  {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())}) -> {msg[:110]} "
                  f"[{time.monotonic() - t:.1f} s]")
        return out

    def ok(self, out: dict) -> bool:
        return out.get("message", "").startswith("OK")

    def scene(self) -> dict:
        return json.loads(self.call("look")["message"])


def _retry(d: Driver, tool: str, tries: int = 2, **args) -> bool:
    return any(d.ok(d.call(tool, **args)) for _ in range(tries))


def stack_station(d: Driver) -> None:
    """The real tower (relay.py's plan): darkest cube is the base, a rod goes in the top cube on the table, then
    middle cube, then the top cube carrying its rod, then the hold."""
    s = d.scene()
    cubes = sorted((o for o in s["objects"] if o["kind"] == "cube"), key=lambda o: o["brightness"])
    colors = list(dict.fromkeys(o["color"] for o in cubes))
    if len(colors) < 2:
        print(f"  only {len(colors)} cube colour(s) in view; can't build a tower")
        return
    base, top, middle = colors[0], colors[-1], colors[1:-1]
    cube_cols = set(colors)
    rods = [o for o in s["objects"] if o["kind"] == "rod"]
    rods.sort(key=lambda o: o["color"] in cube_cols)  # a rod no cube shares the colour of first
    d.call("stack_cube", cube=base, on="tower_spot")
    rod_ok = bool(rods) and _retry(d, "insert_rod", rod=rods[0]["color"], cube=top)
    below = base
    for c in middle + [top]:
        if not _retry(d, "stack_cube", cube=c, on=below):
            print(f"  the {c} cube didn't go on; the tower stops at {below}")
            break
        below = c
    if not rod_ok and rods:
        _retry(d, "insert_rod", rod=rods[0]["color"], cube=below)
    _retry(d, "check_tower")


def drive(url: str, own_color: str = "", seed: int = -1, station: bool = False) -> bool:
    d = Driver(url)
    if not d.mcp.wait_for_ready(timeout=30):
        print(f"no MCP server at {url}")
        return False
    tools = [t["name"] for t in d.mcp.list_tools()]
    print(f"MCP tools: {', '.join(tools)}")
    d.call("start_match", own_color=own_color, seed=seed)
    print("stage 1: unplug")
    for _ in range(TRIES):
        if d.ok(d.call("unplug_cable")):
            break
    print("stage 2: stack")
    if station:
        stack_station(d)
    # base cube -> rod in it -> cube on the rod; a piece that failed twice goes to the back of the line
    fails: dict[str, int] = {}

    def pick(options: list[str]) -> str | None:
        usable = [o for o in options if fails.get(o, 0) < 2 * TRIES]
        return min(usable, key=lambda o: fails.get(o, 0) >= 2, default=None)

    for _ in range(0 if station else TRIES * 4):
        s = d.scene()
        on_tower = [p.split()[1] for p in s["tower"]["pieces"]]  # "cube blue" -> "blue"
        pieces = s["tower"]["pieces"]
        if s["tower"]["height_m"] >= s["tower"].get("pass_height_m", 0.152):
            if d.ok(d.call("check_tower")):
                break
            continue
        free_cubes = [c for c in ("blue", "green", "purple") if not s["cubes"][c]["on_tower"]]
        if not pieces:
            piece, call = pick(free_cubes), lambda p: d.call("stack_cube", cube=p, on="tower_spot")
        elif pieces[-1].startswith("cube"):
            rods = [r for r in ("red", "yellow", "green") if not s["rods"][r]["on_tower"] and s["rods"][r]["upright"]]
            piece, call = pick(rods), lambda p: d.call("insert_rod", rod=p, cube=on_tower[-1])
        else:
            piece, call = pick(free_cubes), lambda p: d.call("stack_cube", cube=p, on=f"rod:{on_tower[-1]}")
        if piece is None:
            print("  out of usable pieces")
            break
        if not d.ok(call(piece)):
            fails[piece] = fails.get(piece, 0) + 1
    status = json.loads(d.call("match_status")["message"])
    if "stack" in status["passed"]:
        print("stage 3: balls")
        own = status["own_color"]
        misses = 0
        while misses < TRIES:
            s = d.scene()
            if station:
                mine = [o for o in s["objects"] if o["kind"] == "ball" and o["color"] == own and not o["in_bin"]]
            else:
                mine = [b for b in s["balls"].values() if b["color"] == own and not b["in_basket"]]
            if not mine:
                break
            if d.ok(d.call("ball_to_bin", color=own)):
                misses = 0
            else:
                misses += 1
    d.call("park_arms")
    status = json.loads(d.call("match_status")["message"])
    print(f"final: score {status['score']}, passed {status['passed']}, bin {status['bin']}, "
          f"{status['elapsed_s']} s")
    return True
