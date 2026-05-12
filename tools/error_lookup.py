"""
tools/error_lookup.py — Error code lookup tool backed by the chunk store
"""
from __future__ import annotations
import re
from typing import Optional

# Built-in error code dictionary (fallback / supplement to RAG)
ERROR_CODES = {
    "E01": {
        "title": "Low Ink Warning",
        "description": "One or more ink cartridges are running low. Replace soon to avoid print quality issues.",
        "action": "1. Open the printer cover. 2. Remove the indicated cartridge. 3. Insert a new cartridge. 4. Run alignment.",
    },
    "E02": {
        "title": "Ink Cartridge Not Detected",
        "description": "The printer cannot detect one or more ink cartridges.",
        "action": "1. Remove cartridge and clean contacts with a dry cloth. 2. Reinsert firmly. 3. Restart printer.",
    },
    "E03": {
        "title": "Ink Cartridge Error",
        "description": "Ink cartridge is incompatible or damaged.",
        "action": "Replace with a compatible OEM cartridge. Avoid third-party refills.",
    },
    "E04": {
        "title": "Paper Feed Error",
        "description": "Paper not feeding correctly from the tray.",
        "action": "1. Remove all paper. 2. Fan the stack to prevent sticking. 3. Reload and align to guides. 4. Retry.",
    },
    "E05": {
        "title": "Paper Jam",
        "description": "Paper is jammed inside the printer mechanism.",
        "action": "1. Turn off printer. 2. Open all covers. 3. Gently pull jammed paper toward you — do not tear. 4. Check rollers for torn pieces. 5. Close covers and restart.",
    },
    "E06": {
        "title": "Output Tray Full",
        "description": "The output tray has reached its maximum capacity.",
        "action": "Remove printed documents from the output tray and resume printing.",
    },
    "E07": {
        "title": "Door Open",
        "description": "A printer door or cover is not properly closed.",
        "action": "Check and firmly close all printer doors and covers.",
    },
    "E08": {
        "title": "Waste Ink Absorber Full",
        "description": "The ink absorber pad has reached its end of life.",
        "action": "Contact authorized service center. Do not attempt self-repair.",
    },
    "E09": {
        "title": "General Print Error",
        "description": "An unspecified print error occurred.",
        "action": "1. Cancel all print jobs. 2. Restart printer and computer. 3. Try printing again.",
    },
    "E10": {
        "title": "Firmware Update Required",
        "description": "Printer firmware is outdated and requires update.",
        "action": "Connect printer to internet and use the manufacturer's update utility.",
    },
    "E11": {
        "title": "Memory Full",
        "description": "Printer memory is full and cannot accept new jobs.",
        "action": "Clear print queue, reduce print job size, or increase virtual memory in settings.",
    },
    "E12": {
        "title": "Network Timeout",
        "description": "Printer lost connection to the network.",
        "action": "1. Check Wi-Fi/Ethernet connection. 2. Restart router and printer. 3. Re-add printer to network.",
    },
    "E13": {
        "title": "Scanner Error",
        "description": "The scanner unit encountered an error.",
        "action": "1. Ensure the scanner glass is clean. 2. Remove any obstructions. 3. Restart printer.",
    },
    "E14": {
        "title": "Alignment Error",
        "description": "Printhead alignment failed.",
        "action": "1. Load plain white paper. 2. Run alignment from printer settings menu. 3. If persistent, clean printhead.",
    },
    "E15": {
        "title": "USB Connection Error",
        "description": "USB connection to the host computer was lost.",
        "action": "1. Try a different USB port/cable. 2. Ensure cable is firmly connected. 3. Reinstall printer driver.",
    },
}


def lookup_error(error_code: Optional[str] = None, query: Optional[str] = None) -> str:
    """
    Look up an error code definition and resolution steps.
    Can accept direct code or extract code from a natural language query.
    """
    code = error_code

    # Try to extract code from query text
    if not code and query:
        match = re.search(r"E\d{2,3}", query.upper())
        if match:
            code = match.group(0)

    if not code:
        # Return all codes summary
        lines = ["Available Error Codes:\n"]
        for c, info in ERROR_CODES.items():
            lines.append(f"  {c}: {info['title']}")
        return "\n".join(lines)

    code = code.upper().strip()
    if code in ERROR_CODES:
        info = ERROR_CODES[code]
        return (
            f"Error Code: {code} — {info['title']}\n\n"
            f"Description: {info['description']}\n\n"
            f"Resolution Steps:\n{info['action']}"
        )
    else:
        return f"Error code '{code}' not found in the lookup database."
