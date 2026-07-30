import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import math

@dataclass(frozen=True)
class Trip:
    trip_id: str
    depart: float
    origin_edge: str
    destination_edge: str
    route_edges: tuple[str, ...]

def load_resco_trips(demand_file: str) -> dict[str, dict]:
    root = ET.parse(demand_file).getroot()
    trips = {}

    for trip in root.findall("trip"):
        trip_id = trip.attrib["id"]

        trips[trip_id] = {
            "trip_id": trip_id,
            "depart": float(trip.attrib["depart"]),
            "origin_edge": trip.attrib["from"],
            "destination_edge": trip.attrib["to"],
        }

    if not trips:
        raise ValueError(f"No <trip> definitions found in {demand_file}")

    return trips

def generate_duarouter_routes(network_file: str, demand_file: str, output_file: str) -> None:
    duarouter = shutil.which("duarouter")

    if duarouter is None:
        raise RuntimeError("Cannot locate duarouter. Ensure SUMO is installed and duarouter is available on PATH.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output_path.with_suffix(".tmp.rou.xml")

    command = [
        duarouter,
        "--net-file",
        os.path.abspath(network_file),
        "--route-files",
        os.path.abspath(demand_file),
        "--output-file",
        str(temporary_output),
        "--routing-algorithm",
        "dijkstra",
        "--seed",
        "42",
    ]

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "duarouter failed while resolving fixed RESCO routes:\n"
            f"{error.stderr}"
        ) from error

    os.replace(temporary_output, output_path)

def load_duarouter_route_edges(duarouter_file: str) -> dict[str, tuple[str, ...]]:
    root = ET.parse(duarouter_file).getroot()
    resolved = {}

    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.attrib["id"]
        route = vehicle.find("route")

        if route is None:
            raise ValueError(
                f"Resolved vehicle {vehicle_id!r} has no route."
            )

        edges = tuple(route.attrib.get("edges", "").split())

        if not edges:
            raise ValueError(
                f"Resolved vehicle {vehicle_id!r} has an empty route."
            )

        resolved[vehicle_id] = edges

    return resolved

def bind_duarouter_routes_to_agents(
    agents,
    origin_edges: tuple[str, ...],
    destination_edges: tuple[str, ...],
    trip_definitions: dict[str, dict],
    resolved_routes: dict[str, tuple[str, ...]],
    departure_offset: float,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """"""