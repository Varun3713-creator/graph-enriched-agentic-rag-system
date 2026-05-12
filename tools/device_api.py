"""
tools/device_api.py — Simulated device status API tool
"""
from __future__ import annotations
import random
from datetime import datetime
from typing import Optional


DEVICE_DB = {
    "printer_001": {
        "model": "LaserJet Pro X500",
        "status": "online",
        "ink_level": 72,
        "paper_tray": "full",
        "last_error": None,
        "uptime_hours": 1247,
    },
    "printer_002": {
        "model": "OfficeJet 8025",
        "status": "error",
        "ink_level": 12,
        "paper_tray": "empty",
        "last_error": "E05 - Paper jam in tray 1",
        "uptime_hours": 834,
    },
    "printer_003": {
        "model": "WorkForce Pro",
        "status": "idle",
        "ink_level": 55,
        "paper_tray": "low",
        "last_error": "E12 - Network timeout",
        "uptime_hours": 2103,
    },
}


def get_device_status(device_id: Optional[str] = None) -> str:
    """
    Return status of a specific device or all devices.
    Simulates a live API call.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if device_id and device_id in DEVICE_DB:
        d = DEVICE_DB[device_id]
        return (
            f"[Device Status @ {timestamp}]\n"
            f"Device: {device_id} ({d['model']})\n"
            f"Status: {d['status'].upper()}\n"
            f"Ink Level: {d['ink_level']}%\n"
            f"Paper Tray: {d['paper_tray']}\n"
            f"Last Error: {d['last_error'] or 'None'}\n"
            f"Uptime: {d['uptime_hours']} hours"
        )

    # Return summary of all devices
    lines = [f"[All Devices Status @ {timestamp}]"]
    for did, d in DEVICE_DB.items():
        lines.append(
            f"  {did} ({d['model']}): {d['status'].upper()} | Ink: {d['ink_level']}% | "
            f"Last Error: {d['last_error'] or 'None'}"
        )
    return "\n".join(lines)
