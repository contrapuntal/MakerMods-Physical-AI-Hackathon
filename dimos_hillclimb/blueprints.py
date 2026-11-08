"""DimOS blueprints for the Hill Climb relay.

- `hillclimb_sim_mcp`: the station's skills behind DimOS's MCP server (port 9990), no LLM. Drive it with
  `dimos mcp call ...`, `python -m dimos_hillclimb drive` (the scripted planner), or any MCP client (Claude Code).
- `hillclimb_sim_agent(model)`: the same plus DimOS's built-in agent (McpClient), which reads the system prompt in
  prompt.py and sequences the skills itself. Send it the go with `dimos agent-send` or `dimos humancli`.
"""

from __future__ import annotations

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect

from dimos_hillclimb.prompt import system_prompt
from dimos_hillclimb.skills import HillClimbStation


def hillclimb_mcp(backend: str = "sim", own_color: str = "orange", seed: int | None = None,
                  time_limit_s: float = 300.0, station_url: str = "http://127.0.0.1:8765",
                  dashboard_host: str = "127.0.0.1", dashboard_port: int = 8770):
    return autoconnect(
        HillClimbStation.blueprint(backend=backend, own_color=own_color, seed=seed, time_limit_s=time_limit_s,
                                   station_url=station_url, dashboard_host=dashboard_host,
                                   dashboard_port=dashboard_port),
        McpServer.blueprint(),
    )


def hillclimb_agent(model: str, backend: str = "sim", own_color: str = "orange", seed: int | None = None,
                    time_limit_s: float = 300.0, station_url: str = "http://127.0.0.1:8765",
                    dashboard_host: str = "127.0.0.1", dashboard_port: int = 8770):
    return autoconnect(
        hillclimb_mcp(backend, own_color, seed, time_limit_s, station_url, dashboard_host, dashboard_port),
        McpClient.blueprint(system_prompt=system_prompt(backend), model=model),
    )


hillclimb_sim_mcp = hillclimb_mcp()
