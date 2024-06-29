import os
import json
import logging

# Configure logging
logging.basicConfig(
    filename="diagnostics.log",  # Name of the log file
    filemode="a",  # Append to the file
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Log level
)


KANTO_MAP_PATH = os.path.join(os.path.dirname(__file__), "kanto_map_dsv.png")
MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}


# Handle KeyErrors
def local_to_global(r: int, c: int, map_n: int):
    try:
        (
            map_x,
            map_y,
        ) = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return 0, 0
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return 0, 0
    

def get_map_name(map_n: int):
    try:
        return MAP_DATA[map_n]["name"]
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return "unknown_map"

with open(MAP_PATH) as f:
    map = json.load(f)
progression_order = [
    "Pallet Town",
    "Route 1",
    "Viridian City",
    "Route 2",
    "Viridian Forest",
    "Pewter City",
    "Route 3",
    "Mt Moon",
    "Route 4",
    "Cerulean City",
    "Route 24",
    "Route 25",
    "Route 5",
    "Route 6",
    "Vermilion City",
    "S.S. Anne",
    "Route 11",
    "Diglett's Cave",
    "Route 9",
    "Route 10",
    "Rock Tunnel",
    "Lavender Town",
    "Route 8",
    "Route 7",
    "Celadon City",
    "Rocket Hideout",
    "Pokemon Tower",
    "Route 12",
    "Route 13",
    "Route 14",
    "Route 15",
    "Fuchsia City",
    "Safari Zone",
    "Sea Route 19",
    "Sea Route 20",
    "Cinnabar Island",
    "Pokemon Mansion",
    "Sea Route 21",
    "Route 22",
    "Victory Road",
    "Indigo Plateau",
]
name_to_id = {region["name"]: int(region["id"]) for region in map["regions"] if region["id"].isdigit()}
ESSENTIAL_MAP_LOCATIONS = {name_to_id[name]: idx for idx, name in enumerate(progression_order) if name in name_to_id}
logging.info(f'ESSENTIAL_MAP_LOCATIONS: {ESSENTIAL_MAP_LOCATIONS}')