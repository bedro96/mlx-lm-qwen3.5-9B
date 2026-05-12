"""
MCP stdio server providing mock factory sensor data.

This server exposes a `get_machine_temperature` tool that returns simulated
temperature readings for factory machines.  It communicates over stdio using
the Model Context Protocol (JSON-RPC).

Usage (standalone test):
    uv run mcp_server.py

In production it is launched as a subprocess by main_mcp.py.
"""

import random
from datetime import datetime, timezone
from typing import TypedDict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("factory-sensors")
_RNG = random.SystemRandom()


class MachineInfo(TypedDict):
    name: str
    zone: str
    normal_range: tuple[float, float]


# Mock machine database — simulates a factory floor with various equipment.
_MACHINES: dict[str, MachineInfo] = {
    "CNC-001": {"name": "CNC Milling Machine #1", "zone": "A", "normal_range": (35, 55)},
    "CNC-002": {"name": "CNC Milling Machine #2", "zone": "A", "normal_range": (35, 55)},
    "WELD-001": {"name": "Robotic Welder #1", "zone": "B", "normal_range": (60, 120)},
    "WELD-002": {"name": "Robotic Welder #2", "zone": "B", "normal_range": (60, 120)},
    "PRESS-001": {"name": "Hydraulic Press #1", "zone": "C", "normal_range": (40, 70)},
    "CONV-001": {"name": "Main Conveyor Belt", "zone": "D", "normal_range": (25, 45)},
    "FURNACE-001": {"name": "Heat Treatment Furnace", "zone": "E", "normal_range": (800, 950)},
    "PUMP-001": {"name": "Coolant Pump #1", "zone": "F", "normal_range": (30, 50)},
    "ROBOT-001": {"name": "Assembly Robot Arm #1", "zone": "G", "normal_range": (30, 55)},
    "COMPRESSOR-001": {"name": "Air Compressor", "zone": "H", "normal_range": (45, 75)},
}


def _generate_reading(machine_id: str) -> dict[str, object]:
    """Generate a mock temperature reading for a given machine."""
    info = _MACHINES.get(machine_id.upper())
    if info is None:
        return {
            "error": f"Unknown machine '{machine_id}'.",
            "available_machines": list(_MACHINES.keys()),
        }

    low, high = info["normal_range"]

    # 80% chance: normal range. 15%: slightly elevated. 5%: critical.
    roll = _RNG.random()
    if roll < 0.80:
        temp = round(_RNG.uniform(low, high), 1)
        status = "NORMAL"
    elif roll < 0.95:
        temp = round(high + _RNG.uniform(1, high * 0.15), 1)
        status = "WARNING"
    else:
        temp = round(high + _RNG.uniform(high * 0.15, high * 0.35), 1)
        status = "CRITICAL"

    return {
        "machine_id": machine_id.upper(),
        "machine_name": info["name"],
        "zone": info["zone"],
        "temperature_celsius": temp,
        "normal_range": f"{low}–{high} °C",
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_machine_temperature(machine_id: str) -> dict[str, object]:
    """Get the current temperature reading for a specific factory machine.

    Args:
        machine_id: The machine identifier, e.g. 'CNC-001', 'WELD-002', 'FURNACE-001'.
                    Use 'ALL' to get readings for every machine on the factory floor.
    """
    if machine_id.upper() == "ALL":
        readings = [_generate_reading(mid) for mid in _MACHINES]
        return {"readings": readings, "total_machines": len(readings)}
    return _generate_reading(machine_id)


@mcp.tool()
def list_machines() -> dict[str, object]:
    """List all available machines with their zones and normal temperature ranges."""
    machines = []
    for mid, info in _MACHINES.items():
        low, high = info["normal_range"]
        machines.append(
            {
                "machine_id": mid,
                "name": info["name"],
                "zone": info["zone"],
                "normal_temp_range": f"{low}–{high} °C",
            }
        )
    return {"machines": machines, "total": len(machines)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
