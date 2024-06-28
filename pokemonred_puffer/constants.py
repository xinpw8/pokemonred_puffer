from pyboy.utils import WindowEvent
from gymnasium import spaces
import numpy as np
from enum import Enum


STATES_TO_SAVE_LOAD = [
    "recent_frames",
    "agent_stats",
    "base_explore",
    "max_opponent_level",
    "max_event_rew",
    "max_level_rew",
    "last_health",
    "last_num_poke",
    "last_num_mon_in_box",
    "total_healing_rew",
    "died_count",
    "prev_knn_rew",
    "visited_pokecenter_list",
    "last_10_map_ids",
    "last_10_coords",
    "past_events_string",
    "last_10_event_ids",
    "early_done",
    "step_count",
    "past_rewards",
    "base_event_flags",
    "rewarded_events_string",
    "seen_map_dict",
    "_cut_badge",
    "_have_hm01",
    "_can_use_cut",
    "_surf_badge",
    "_have_hm03",
    "_can_use_surf",
    "_have_pokeflute",
    "_have_silph_scope",
    "used_cut_coords_dict",
    "_last_item_count",
    "_is_box_mon_higher_level",
    "hideout_elevator_maps",
    "use_mart_count",
    "use_pc_swap_count",
    "seen_coords",
    "perm_seen_coords",
    "special_seen_coords_count",
    "secret_switch_states",
    "party_level_base",
    "party_level_post",
]

GYM_INFO = [
    {
        "badge": 0,
        "num_poke": 2,
        "max_level": 14,
    },
    {
        "badge": 1,
        "num_poke": 2,
        "max_level": 21,
    },
    {
        "badge": 2,
        "num_poke": 3,
        "max_level": 24,
    },
    {
        "badge": 3,
        "num_poke": 3,
        "max_level": 29,
    },
    {
        "badge": 4,
        "num_poke": 4,
        "max_level": 43,
    },
    {
        "badge": 5,
        "num_poke": 4,
        "max_level": 43,
    },
    {
        "badge": 6,
        "num_poke": 4,
        "max_level": 47,
    },
    {
        "badge": 7,
        "num_poke": 6,
        "max_level": 100,
    },
    # {
    #     'badge': 7,
    #     'num_poke': 5,
    #     'max_level': 50,
    # },
    # {
    #     'badge': 8,
    #     'num_poke': 5,
    #     'max_level': 56,
    # },
    # {
    #     'badge': 9,
    #     'num_poke': 5,
    #     'max_level': 58,
    # },
    # {
    #     'badge': 10,
    #     'num_poke': 5,
    #     'max_level': 60,
    # },
    # {
    #     'badge': 11,
    #     'num_poke': 5,
    #     'max_level': 62,
    # },
    {
        "badge": 12,
        "num_poke": 6,
        "max_level": 100,
    },
]

CAVE_MAP_IDS = [
    0x3B,  # Mt. Moon 1F
    0x3C,  # Mt. Moon B1F
    0x3D,  # Mt. Moon B2F
    0x52,  # Rock Tunnel 1F
    0xC5,  # DIGLETTS_CAVE
    0xE8,  # ROCK_TUNNEL_B1F
    0x6C,  # VICTORY_ROAD_1F
    0xC6,  # VICTORY_ROAD_3F
    0xC2,  # VICTORY_ROAD_2F
    0xE2,  # CERULEAN_CAVE_2F
    0xE4,  # CERULEAN_CAVE_1F
    0xE3,  # CERULEAN_CAVE_B1F
    0xC0,  # SEAFOAM_ISLANDS_1F
    0x9F,  # SEAFOAM_ISLANDS_B1F
    0xA0,  # SEAFOAM_ISLANDS_B2F
    0xA1,  # SEAFOAM_ISLANDS_B3F
    0xA2,  # SEAFOAM_ISLANDS_B4F
    0x2E,  # DIGLETTS_CAVE_ROUTE_2
    0x55,  # DIGLETTS_CAVE_ROUTE_11
]

GYM_MAP_IDS = [
    0x36,  # Pewter City Gym
    0x41,  # Cerulean City Gym
    0x5C,  # Vermilion City Gym
    0x86,  # Celadon City Gym
    0x9D,  # Fuchsia City Gym
    0xB2,  # Saffron City Gym
    0xA6,  # Cinnabar Island Gym
    0x2D,  # Viridian City Gym
    # Elite Four
    0xF5,  # Lorelei Room
    0xF6,  # Bruno Room
    0xF7,  # Agatha Room
    0x71,  # Lance Room
    0x78,  # Champion Room
]

ETC_MAP_IDS = [
    # SS Anne
    0x5F,  # 1F
    0x60,  # 2F
    0x61,  # 3F
    0x62,  # B1F
    0x63,  # SS_ANNE_BOW
    0x64,  # SS_ANNE_KITCHEN
    0x65,  # SS_ANNE_CAPTAINS_ROOM
    0x66,  # SS_ANNE_1F_ROOMS
    0x67,  # SS_ANNE_2F_ROOMS
    0x68,  # SS_ANNE_B1F_ROOMS
    # Power Plant
    0x53,  # Power Plant
    # Rocket Hideout
    0x87,  # GAME_CORNER
    0xC7,  # ROCKET_HIDEOUT_B1F
    0xC8,  # ROCKET_HIDEOUT_B2F
    0xC9,  # ROCKET_HIDEOUT_B3F
    0xCA,  # ROCKET_HIDEOUT_B4F
    0xCB,  # ROCKET_HIDEOUT_ELEVATOR
    # Silph Co
    0xB5,  # SILPH_CO_1F
    0xCF,  # SILPH_CO_2F
    0xD0,  # SILPH_CO_3F
    0xD1,  # SILPH_CO_4F
    0xD2,  # SILPH_CO_5F
    0xD3,  # SILPH_CO_6F
    0xD4,  # SILPH_CO_7F
    0xD5,  # SILPH_CO_8F
    0xE9,  # SILPH_CO_9F
    0xEA,  # SILPH_CO_10F
    0xEB,  # SILPH_CO_11F
    0xEC,  # SILPH_CO_ELEVATOR
    # Pokémon Tower
    0x8E,  # POKEMON_TOWER_1F
    0x8F,  # POKEMON_TOWER_2F
    0x90,  # POKEMON_TOWER_3F
    0x91,  # POKEMON_TOWER_4F
    0x92,  # POKEMON_TOWER_5F
    0x93,  # POKEMON_TOWER_6F
    0x94,  # POKEMON_TOWER_7F
    # Pokémon Mansion
    0xA5,  # POKEMON_MANSION_1F
    0xD6,  # POKEMON_MANSION_2F
    0xD7,  # POKEMON_MANSION_3F
    0xD8,  # POKEMON_MANSION_B1F
    # Safari Zone
    0x9C,  # SAFARI_ZONE_GATE
    0xD9,  # SAFARI_ZONE_EAST
    0xDA,  # SAFARI_ZONE_NORTH
    0xDB,  # SAFARI_ZONE_WEST
    0xDC,  # SAFARI_ZONE_CENTER
    0xDD,  # SAFARI_ZONE_CENTER_REST_HOUSE
    0xDE,  # SAFARI_ZONE_SECRET_HOUSE
    0xDF,  # SAFARI_ZONE_WEST_REST_HOUSE
    0xE0,  # SAFARI_ZONE_EAST_REST_HOUSE
    0xE1,  # SAFARI_ZONE_NORTH_REST_HOUSE
    # Sea routes
    0x1E,  # ROUTE_19
    0x1F,  # ROUTE_20
    0x20,  # ROUTE_21
]

SPECIAL_MAP_IDS = [] + CAVE_MAP_IDS + GYM_MAP_IDS + ETC_MAP_IDS


IGNORED_EVENT_IDS = [
    30,  # enter town map house
    29,  # leave town map house
    111,  # museum ticket
    1314,  # route 22 first rival battle
    1016,  # magikrap trade in Mt Moon Pokecenter
]


SPECIAL_KEY_ITEM_IDS = [
    0x30,  # CARD_KEY
    0x2B,  # SECRET_KEY
    0x48,  # SILPH_SCOPE
    0x4A,  # LIFT_KEY
    0x49,  # POKE_FLUTE
    0x3F,  # S_S_TICKET
    0x06,  # BICYCLE
    0x40,  # GOLD_TEETH
    0x3C,  # FRESH_WATER
    # 0x3D,  # SODA_POP
    # 0x3E,  # LEMONADE
]


ALL_KEY_ITEMS = [
    0x05,  # TOWN_MAP
    0x06,  # BICYCLE
    0x07,  # SURFBOARD
    0x08,  # SAFARI_BALL
    0x09,  # POKEDEX
    0x15,  # BOULDERBADGE
    0x16,  # CASCADEBADGE
    0x17,  # THUNDERBADGE
    0x18,  # RAINBOWBADGE
    0x19,  # SOULBADGE
    0x1A,  # MARSHBADGE
    0x1B,  # VOLCANOBADGE
    0x1C,  # EARTHBADGE
    0x20,  # OLD_AMBER
    0x29,  # DOME_FOSSIL
    0x2A,  # HELIX_FOSSIL
    0x2B,  # SECRET_KEY
    0x2D,  # BIKE_VOUCHER
    0x30,  # CARD_KEY
    0x3F,  # S_S_TICKET
    0x40,  # GOLD_TEETH
    0x45,  # COIN_CASE
    0x46,  # OAKS_PARCEL
    0x47,  # ITEMFINDER
    0x48,  # SILPH_SCOPE
    0x49,  # POKE_FLUTE
    0x4A,  # LIFT_KEY
    0x4C,  # OLD_ROD
    0x4D,  # GOOD_ROD
    0x4E,  # SUPER_ROD
    # quest items to keep
    0x3C,  # FRESH_WATER
    # 0x3D,  # SODA_POP
    # 0x3E,  # LEMONADE
]

ALL_HM_IDS = [
    0xC4,  # CUT
    0xC5,  # FLY
    0xC6,  # SURF
    0xC7,  # STRENGTH
    0xC8,  # FLASH
]

STRENGTH = 0x46

ALL_POKEBALL_IDS = [
    0x01,  # MASTER_BALL
    0x02,  # ULTRA_BALL
    0x03,  # GREAT_BALL
    0x04,  # POKE_BALL
]

# const FULL_RESTORE  ; $10
# const MAX_POTION    ; $11
# const HYPER_POTION  ; $12
# const SUPER_POTION  ; $13
# const POTION        ; $14
# const FULL_HEAL     ; $34
# const REVIVE        ; $35
# const MAX_REVIVE    ; $36
# const ELIXER        ; $52
# const MAX_ELIXER    ; $53
ALL_HEALABLE_ITEM_IDS = [  # from worst to best, so that it will consume the worst first
    0x14,  # POTION
    0x13,  # SUPER_POTION
    0x12,  # HYPER_POTION
    0x11,  # MAX_POTION
    0x35,  # REVIVE
    0x34,  # FULL_HEAL
    0x10,  # FULL_RESTORE
    0x36,  # MAX_REVIVE
    0x52,  # ELIXER
    0x53,  # MAX_ELIXER
]

ALL_GOOD_ITEMS = ALL_KEY_ITEMS + ALL_POKEBALL_IDS + ALL_HEALABLE_ITEM_IDS + ALL_HM_IDS

GOOD_ITEMS_PRIORITY = [  # from worst to best, so that it will toss the worst first
    0x04,  # POKE_BALL
    0x14,  # POTION
    0x03,  # GREAT_BALL
    0x13,  # SUPER_POTION
    0x12,  # HYPER_POTION
    0x02,  # ULTRA_BALL
    0x35,  # REVIVE
    0x50,  # ETHER
    0x51,  # MAX_ETHER
    0x01,  # MASTER_BALL
    0x11,  # MAX_POTION
    0x34,  # FULL_HEAL
    0x52,  # ELIXER
    0x53,  # MAX_ELIXER
    0x10,  # FULL_RESTORE
    0x36,  # MAX_REVIVE
]

POKEBALL_PRIORITY = [
    # 0x01,  # MASTER_BALL  # not purchasable
    0x02,  # ULTRA_BALL
    0x03,  # GREAT_BALL
    0x04,  # POKE_BALL
]

POTION_PRIORITY = [
    0x10,  # FULL_RESTORE
    0x11,  # MAX_POTION
    0x12,  # HYPER_POTION
    0x13,  # SUPER_POTION
    0x14,  # POTION
]

REVIVE_PRIORITY = [
    0x36,  # MAX_REVIVE
    0x35,  # REVIVE
]

LEVELS = [
    {  # 0 level 1
        "last_pokecenter": ["VERMILION_CITY", "LAVENDER_TOWN"]
    },
    {  # 1 level 2
        "last_pokecenter": ["CELADON_CITY"],
    },
    {  # 2 level 3
        "last_pokecenter": ["LAVENDER_TOWN"],
    },
    {  # 3 level 4
        "badge": 5,
    },
    {  # 4 level 5
        "last_pokecenter": ["FUCHSIA_CITY"],
    },
    {  # 5 level 6
        "last_pokecenter": ["CINNABAR_ISLAND"],
    },
    {  # 6 level 7
        "last_pokecenter": ["VIRIDIAN_CITY"],
    },
    {  # 7 level 8
        "event": "CHAMPION",
    },
]

MAP_DICT = {
    "PALLET_TOWN": {"width": 20, "height": 18, "map_hex": "00", "map_id": 0},
    "VIRIDIAN_CITY": {"width": 40, "height": 36, "map_hex": "01", "map_id": 1},
    "PEWTER_CITY": {"width": 40, "height": 36, "map_hex": "02", "map_id": 2},
    "CERULEAN_CITY": {"width": 40, "height": 36, "map_hex": "03", "map_id": 3},
    "LAVENDER_TOWN": {"width": 20, "height": 18, "map_hex": "04", "map_id": 4},
    "VERMILION_CITY": {"width": 40, "height": 36, "map_hex": "05", "map_id": 5},
    "CELADON_CITY": {"width": 50, "height": 36, "map_hex": "06", "map_id": 6},
    "FUCHSIA_CITY": {"width": 40, "height": 36, "map_hex": "07", "map_id": 7},
    "CINNABAR_ISLAND": {"width": 20, "height": 18, "map_hex": "08", "map_id": 8},
    "INDIGO_PLATEAU": {"width": 20, "height": 18, "map_hex": "09", "map_id": 9},
    "SAFFRON_CITY": {"width": 40, "height": 36, "map_hex": "0A", "map_id": 10},
    "UNUSED_MAP_0B": {"width": 0, "height": 0, "map_hex": "0B", "map_id": 11},
    "ROUTE_1": {"width": 20, "height": 36, "map_hex": "0C", "map_id": 12},
    "ROUTE_2": {"width": 20, "height": 72, "map_hex": "0D", "map_id": 13},
    "ROUTE_3": {"width": 70, "height": 18, "map_hex": "0E", "map_id": 14},
    "ROUTE_4": {"width": 90, "height": 18, "map_hex": "0F", "map_id": 15},
    "ROUTE_5": {"width": 20, "height": 36, "map_hex": "10", "map_id": 16},
    "ROUTE_6": {"width": 20, "height": 36, "map_hex": "11", "map_id": 17},
    "ROUTE_7": {"width": 20, "height": 18, "map_hex": "12", "map_id": 18},
    "ROUTE_8": {"width": 60, "height": 18, "map_hex": "13", "map_id": 19},
    "ROUTE_9": {"width": 60, "height": 18, "map_hex": "14", "map_id": 20},
    "ROUTE_10": {"width": 20, "height": 72, "map_hex": "15", "map_id": 21},
    "ROUTE_11": {"width": 60, "height": 18, "map_hex": "16", "map_id": 22},
    "ROUTE_12": {"width": 20, "height": 108, "map_hex": "17", "map_id": 23},
    "ROUTE_13": {"width": 60, "height": 18, "map_hex": "18", "map_id": 24},
    "ROUTE_14": {"width": 20, "height": 54, "map_hex": "19", "map_id": 25},
    "ROUTE_15": {"width": 60, "height": 18, "map_hex": "1A", "map_id": 26},
    "ROUTE_16": {"width": 40, "height": 18, "map_hex": "1B", "map_id": 27},
    "ROUTE_17": {"width": 20, "height": 144, "map_hex": "1C", "map_id": 28},
    "ROUTE_18": {"width": 50, "height": 18, "map_hex": "1D", "map_id": 29},
    "ROUTE_19": {"width": 20, "height": 54, "map_hex": "1E", "map_id": 30},
    "ROUTE_20": {"width": 100, "height": 18, "map_hex": "1F", "map_id": 31},
    "ROUTE_21": {"width": 20, "height": 90, "map_hex": "20", "map_id": 32},
    "ROUTE_22": {"width": 40, "height": 18, "map_hex": "21", "map_id": 33},
    "ROUTE_23": {"width": 20, "height": 144, "map_hex": "22", "map_id": 34},
    "ROUTE_24": {"width": 20, "height": 36, "map_hex": "23", "map_id": 35},
    "ROUTE_25": {"width": 60, "height": 18, "map_hex": "24", "map_id": 36},
    "REDS_HOUSE_1F": {"width": 8, "height": 8, "map_hex": "25", "map_id": 37},
    "REDS_HOUSE_2F": {"width": 8, "height": 8, "map_hex": "26", "map_id": 38},
    "BLUES_HOUSE": {"width": 8, "height": 8, "map_hex": "27", "map_id": 39},
    "OAKS_LAB": {"width": 10, "height": 12, "map_hex": "28", "map_id": 40},
    "VIRIDIAN_POKECENTER": {"width": 14, "height": 8, "map_hex": "29", "map_id": 41},
    "VIRIDIAN_MART": {"width": 8, "height": 8, "map_hex": "2A", "map_id": 42},
    "VIRIDIAN_SCHOOL_HOUSE": {"width": 8, "height": 8, "map_hex": "2B", "map_id": 43},
    "VIRIDIAN_NICKNAME_HOUSE": {"width": 8, "height": 8, "map_hex": "2C", "map_id": 44},
    "VIRIDIAN_GYM": {"width": 20, "height": 18, "map_hex": "2D", "map_id": 45},
    "DIGLETTS_CAVE_ROUTE_2": {"width": 8, "height": 8, "map_hex": "2E", "map_id": 46},
    "VIRIDIAN_FOREST_NORTH_GATE": {"width": 10, "height": 8, "map_hex": "2F", "map_id": 47},
    "ROUTE_2_TRADE_HOUSE": {"width": 8, "height": 8, "map_hex": "30", "map_id": 48},
    "ROUTE_2_GATE": {"width": 10, "height": 8, "map_hex": "31", "map_id": 49},
    "VIRIDIAN_FOREST_SOUTH_GATE": {"width": 10, "height": 8, "map_hex": "32", "map_id": 50},
    "VIRIDIAN_FOREST": {"width": 34, "height": 48, "map_hex": "33", "map_id": 51},
    "MUSEUM_1F": {"width": 20, "height": 8, "map_hex": "34", "map_id": 52},
    "MUSEUM_2F": {"width": 14, "height": 8, "map_hex": "35", "map_id": 53},
    "PEWTER_GYM": {"width": 10, "height": 14, "map_hex": "36", "map_id": 54},
    "PEWTER_NIDORAN_HOUSE": {"width": 8, "height": 8, "map_hex": "37", "map_id": 55},
    "PEWTER_MART": {"width": 8, "height": 8, "map_hex": "38", "map_id": 56},
    "PEWTER_SPEECH_HOUSE": {"width": 8, "height": 8, "map_hex": "39", "map_id": 57},
    "PEWTER_POKECENTER": {"width": 14, "height": 8, "map_hex": "3A", "map_id": 58},
    "MT_MOON_1F": {"width": 40, "height": 36, "map_hex": "3B", "map_id": 59},
    "MT_MOON_B1F": {"width": 28, "height": 28, "map_hex": "3C", "map_id": 60},
    "MT_MOON_B2F": {"width": 40, "height": 36, "map_hex": "3D", "map_id": 61},
    "CERULEAN_TRASHED_HOUSE": {"width": 8, "height": 8, "map_hex": "3E", "map_id": 62},
    "CERULEAN_TRADE_HOUSE": {"width": 8, "height": 8, "map_hex": "3F", "map_id": 63},
    "CERULEAN_POKECENTER": {"width": 14, "height": 8, "map_hex": "40", "map_id": 64},
    "CERULEAN_GYM": {"width": 10, "height": 14, "map_hex": "41", "map_id": 65},
    "BIKE_SHOP": {"width": 8, "height": 8, "map_hex": "42", "map_id": 66},
    "CERULEAN_MART": {"width": 8, "height": 8, "map_hex": "43", "map_id": 67},
    "MT_MOON_POKECENTER": {"width": 14, "height": 8, "map_hex": "44", "map_id": 68},
    "CERULEAN_TRASHED_HOUSE_COPY": {"width": 8, "height": 8, "map_hex": "45", "map_id": 69},
    "ROUTE_5_GATE": {"width": 8, "height": 6, "map_hex": "46", "map_id": 70},
    "UNDERGROUND_PATH_ROUTE_5": {"width": 8, "height": 8, "map_hex": "47", "map_id": 71},
    "DAYCARE": {"width": 8, "height": 8, "map_hex": "48", "map_id": 72},
    "ROUTE_6_GATE": {"width": 8, "height": 6, "map_hex": "49", "map_id": 73},
    "UNDERGROUND_PATH_ROUTE_6": {"width": 8, "height": 8, "map_hex": "4A", "map_id": 74},
    "UNDERGROUND_PATH_ROUTE_6_COPY": {"width": 8, "height": 8, "map_hex": "4B", "map_id": 75},
    "ROUTE_7_GATE": {"width": 6, "height": 8, "map_hex": "4C", "map_id": 76},
    "UNDERGROUND_PATH_ROUTE_7": {"width": 8, "height": 8, "map_hex": "4D", "map_id": 77},
    "UNDERGROUND_PATH_ROUTE_7_COPY": {"width": 8, "height": 8, "map_hex": "4E", "map_id": 78},
    "ROUTE_8_GATE": {"width": 6, "height": 8, "map_hex": "4F", "map_id": 79},
    "UNDERGROUND_PATH_ROUTE_8": {"width": 8, "height": 8, "map_hex": "50", "map_id": 80},
    "ROCK_TUNNEL_POKECENTER": {"width": 14, "height": 8, "map_hex": "51", "map_id": 81},
    "ROCK_TUNNEL_1F": {"width": 40, "height": 36, "map_hex": "52", "map_id": 82},
    "POWER_PLANT": {"width": 40, "height": 36, "map_hex": "53", "map_id": 83},
    "ROUTE_11_GATE_1F": {"width": 8, "height": 10, "map_hex": "54", "map_id": 84},
    "DIGLETTS_CAVE_ROUTE_11": {"width": 8, "height": 8, "map_hex": "55", "map_id": 85},
    "ROUTE_11_GATE_2F": {"width": 8, "height": 8, "map_hex": "56", "map_id": 86},
    "ROUTE_12_GATE_1F": {"width": 10, "height": 8, "map_hex": "57", "map_id": 87},
    "BILLS_HOUSE": {"width": 8, "height": 8, "map_hex": "58", "map_id": 88},
    "VERMILION_POKECENTER": {"width": 14, "height": 8, "map_hex": "59", "map_id": 89},
    "POKEMON_FAN_CLUB": {"width": 8, "height": 8, "map_hex": "5A", "map_id": 90},
    "VERMILION_MART": {"width": 8, "height": 8, "map_hex": "5B", "map_id": 91},
    "VERMILION_GYM": {"width": 10, "height": 18, "map_hex": "5C", "map_id": 92},
    "VERMILION_PIDGEY_HOUSE": {"width": 8, "height": 8, "map_hex": "5D", "map_id": 93},
    "VERMILION_DOCK": {"width": 28, "height": 12, "map_hex": "5E", "map_id": 94},
    "SS_ANNE_1F": {"width": 40, "height": 18, "map_hex": "5F", "map_id": 95},
    "SS_ANNE_2F": {"width": 40, "height": 18, "map_hex": "60", "map_id": 96},
    "SS_ANNE_3F": {"width": 20, "height": 6, "map_hex": "61", "map_id": 97},
    "SS_ANNE_B1F": {"width": 30, "height": 8, "map_hex": "62", "map_id": 98},
    "SS_ANNE_BOW": {"width": 20, "height": 14, "map_hex": "63", "map_id": 99},
    "SS_ANNE_KITCHEN": {"width": 14, "height": 16, "map_hex": "64", "map_id": 100},
    "SS_ANNE_CAPTAINS_ROOM": {"width": 6, "height": 8, "map_hex": "65", "map_id": 101},
    "SS_ANNE_1F_ROOMS": {"width": 24, "height": 16, "map_hex": "66", "map_id": 102},
    "SS_ANNE_2F_ROOMS": {"width": 24, "height": 16, "map_hex": "67", "map_id": 103},
    "SS_ANNE_B1F_ROOMS": {"width": 24, "height": 16, "map_hex": "68", "map_id": 104},
    "UNUSED_MAP_69": {"width": 0, "height": 0, "map_hex": "69", "map_id": 105},
    "UNUSED_MAP_6A": {"width": 0, "height": 0, "map_hex": "6A", "map_id": 106},
    "UNUSED_MAP_6B": {"width": 0, "height": 0, "map_hex": "6B", "map_id": 107},
    "VICTORY_ROAD_1F": {"width": 20, "height": 18, "map_hex": "6C", "map_id": 108},
    "UNUSED_MAP_6D": {"width": 0, "height": 0, "map_hex": "6D", "map_id": 109},
    "UNUSED_MAP_6E": {"width": 0, "height": 0, "map_hex": "6E", "map_id": 110},
    "UNUSED_MAP_6F": {"width": 0, "height": 0, "map_hex": "6F", "map_id": 111},
    "UNUSED_MAP_70": {"width": 0, "height": 0, "map_hex": "70", "map_id": 112},
    "LANCES_ROOM": {"width": 26, "height": 26, "map_hex": "71", "map_id": 113},
    "UNUSED_MAP_72": {"width": 0, "height": 0, "map_hex": "72", "map_id": 114},
    "UNUSED_MAP_73": {"width": 0, "height": 0, "map_hex": "73", "map_id": 115},
    "UNUSED_MAP_74": {"width": 0, "height": 0, "map_hex": "74", "map_id": 116},
    "UNUSED_MAP_75": {"width": 0, "height": 0, "map_hex": "75", "map_id": 117},
    "HALL_OF_FAME": {"width": 10, "height": 8, "map_hex": "76", "map_id": 118},
    "UNDERGROUND_PATH_NORTH_SOUTH": {"width": 8, "height": 48, "map_hex": "77", "map_id": 119},
    "CHAMPIONS_ROOM": {"width": 8, "height": 8, "map_hex": "78", "map_id": 120},
    "UNDERGROUND_PATH_WEST_EAST": {"width": 50, "height": 8, "map_hex": "79", "map_id": 121},
    "CELADON_MART_1F": {"width": 20, "height": 8, "map_hex": "7A", "map_id": 122},
    "CELADON_MART_2F": {"width": 20, "height": 8, "map_hex": "7B", "map_id": 123},
    "CELADON_MART_3F": {"width": 20, "height": 8, "map_hex": "7C", "map_id": 124},
    "CELADON_MART_4F": {"width": 20, "height": 8, "map_hex": "7D", "map_id": 125},
    "CELADON_MART_ROOF": {"width": 20, "height": 8, "map_hex": "7E", "map_id": 126},
    "CELADON_MART_ELEVATOR": {"width": 4, "height": 4, "map_hex": "7F", "map_id": 127},
    "CELADON_MANSION_1F": {"width": 8, "height": 12, "map_hex": "80", "map_id": 128},
    "CELADON_MANSION_2F": {"width": 8, "height": 12, "map_hex": "81", "map_id": 129},
    "CELADON_MANSION_3F": {"width": 8, "height": 12, "map_hex": "82", "map_id": 130},
    "CELADON_MANSION_ROOF": {"width": 8, "height": 12, "map_hex": "83", "map_id": 131},
    "CELADON_MANSION_ROOF_HOUSE": {"width": 8, "height": 8, "map_hex": "84", "map_id": 132},
    "CELADON_POKECENTER": {"width": 14, "height": 8, "map_hex": "85", "map_id": 133},
    "CELADON_GYM": {"width": 10, "height": 18, "map_hex": "86", "map_id": 134},
    "GAME_CORNER": {"width": 20, "height": 18, "map_hex": "87", "map_id": 135},
    "CELADON_MART_5F": {"width": 20, "height": 8, "map_hex": "88", "map_id": 136},
    "GAME_CORNER_PRIZE_ROOM": {"width": 10, "height": 8, "map_hex": "89", "map_id": 137},
    "CELADON_DINER": {"width": 10, "height": 8, "map_hex": "8A", "map_id": 138},
    "CELADON_CHIEF_HOUSE": {"width": 8, "height": 8, "map_hex": "8B", "map_id": 139},
    "CELADON_HOTEL": {"width": 14, "height": 8, "map_hex": "8C", "map_id": 140},
    "LAVENDER_POKECENTER": {"width": 14, "height": 8, "map_hex": "8D", "map_id": 141},
    "POKEMON_TOWER_1F": {"width": 20, "height": 18, "map_hex": "8E", "map_id": 142},
    "POKEMON_TOWER_2F": {"width": 20, "height": 18, "map_hex": "8F", "map_id": 143},
    "POKEMON_TOWER_3F": {"width": 20, "height": 18, "map_hex": "90", "map_id": 144},
    "POKEMON_TOWER_4F": {"width": 20, "height": 18, "map_hex": "91", "map_id": 145},
    "POKEMON_TOWER_5F": {"width": 20, "height": 18, "map_hex": "92", "map_id": 146},
    "POKEMON_TOWER_6F": {"width": 20, "height": 18, "map_hex": "93", "map_id": 147},
    "POKEMON_TOWER_7F": {"width": 20, "height": 18, "map_hex": "94", "map_id": 148},
    "MR_FUJIS_HOUSE": {"width": 8, "height": 8, "map_hex": "95", "map_id": 149},
    "LAVENDER_MART": {"width": 8, "height": 8, "map_hex": "96", "map_id": 150},
    "LAVENDER_CUBONE_HOUSE": {"width": 8, "height": 8, "map_hex": "97", "map_id": 151},
    "FUCHSIA_MART": {"width": 8, "height": 8, "map_hex": "98", "map_id": 152},
    "FUCHSIA_BILLS_GRANDPAS_HOUSE": {"width": 8, "height": 8, "map_hex": "99", "map_id": 153},
    "FUCHSIA_POKECENTER": {"width": 14, "height": 8, "map_hex": "9A", "map_id": 154},
    "WARDENS_HOUSE": {"width": 10, "height": 8, "map_hex": "9B", "map_id": 155},
    "SAFARI_ZONE_GATE": {"width": 8, "height": 6, "map_hex": "9C", "map_id": 156},
    "FUCHSIA_GYM": {"width": 10, "height": 18, "map_hex": "9D", "map_id": 157},
    "FUCHSIA_MEETING_ROOM": {"width": 14, "height": 8, "map_hex": "9E", "map_id": 158},
    "SEAFOAM_ISLANDS_B1F": {"width": 30, "height": 18, "map_hex": "9F", "map_id": 159},
    "SEAFOAM_ISLANDS_B2F": {"width": 30, "height": 18, "map_hex": "A0", "map_id": 160},
    "SEAFOAM_ISLANDS_B3F": {"width": 30, "height": 18, "map_hex": "A1", "map_id": 161},
    "SEAFOAM_ISLANDS_B4F": {"width": 30, "height": 18, "map_hex": "A2", "map_id": 162},
    "VERMILION_OLD_ROD_HOUSE": {"width": 8, "height": 8, "map_hex": "A3", "map_id": 163},
    "FUCHSIA_GOOD_ROD_HOUSE": {"width": 8, "height": 8, "map_hex": "A4", "map_id": 164},
    "POKEMON_MANSION_1F": {"width": 30, "height": 28, "map_hex": "A5", "map_id": 165},
    "CINNABAR_GYM": {"width": 20, "height": 18, "map_hex": "A6", "map_id": 166},
    "CINNABAR_LAB": {"width": 18, "height": 8, "map_hex": "A7", "map_id": 167},
    "CINNABAR_LAB_TRADE_ROOM": {"width": 8, "height": 8, "map_hex": "A8", "map_id": 168},
    "CINNABAR_LAB_METRONOME_ROOM": {"width": 8, "height": 8, "map_hex": "A9", "map_id": 169},
    "CINNABAR_LAB_FOSSIL_ROOM": {"width": 8, "height": 8, "map_hex": "AA", "map_id": 170},
    "CINNABAR_POKECENTER": {"width": 14, "height": 8, "map_hex": "AB", "map_id": 171},
    "CINNABAR_MART": {"width": 8, "height": 8, "map_hex": "AC", "map_id": 172},
    "CINNABAR_MART_COPY": {"width": 8, "height": 8, "map_hex": "AD", "map_id": 173},
    "INDIGO_PLATEAU_LOBBY": {"width": 16, "height": 12, "map_hex": "AE", "map_id": 174},
    "COPYCATS_HOUSE_1F": {"width": 8, "height": 8, "map_hex": "AF", "map_id": 175},
    "COPYCATS_HOUSE_2F": {"width": 8, "height": 8, "map_hex": "B0", "map_id": 176},
    "FIGHTING_DOJO": {"width": 10, "height": 12, "map_hex": "B1", "map_id": 177},
    "SAFFRON_GYM": {"width": 20, "height": 18, "map_hex": "B2", "map_id": 178},
    "SAFFRON_PIDGEY_HOUSE": {"width": 8, "height": 8, "map_hex": "B3", "map_id": 179},
    "SAFFRON_MART": {"width": 8, "height": 8, "map_hex": "B4", "map_id": 180},
    "SILPH_CO_1F": {"width": 30, "height": 18, "map_hex": "B5", "map_id": 181},
    "SAFFRON_POKECENTER": {"width": 14, "height": 8, "map_hex": "B6", "map_id": 182},
    "MR_PSYCHICS_HOUSE": {"width": 8, "height": 8, "map_hex": "B7", "map_id": 183},
    "ROUTE_15_GATE_1F": {"width": 8, "height": 10, "map_hex": "B8", "map_id": 184},
    "ROUTE_15_GATE_2F": {"width": 8, "height": 8, "map_hex": "B9", "map_id": 185},
    "ROUTE_16_GATE_1F": {"width": 8, "height": 14, "map_hex": "BA", "map_id": 186},
    "ROUTE_16_GATE_2F": {"width": 8, "height": 8, "map_hex": "BB", "map_id": 187},
    "ROUTE_16_FLY_HOUSE": {"width": 8, "height": 8, "map_hex": "BC", "map_id": 188},
    "ROUTE_12_SUPER_ROD_HOUSE": {"width": 8, "height": 8, "map_hex": "BD", "map_id": 189},
    "ROUTE_18_GATE_1F": {"width": 8, "height": 10, "map_hex": "BE", "map_id": 190},
    "ROUTE_18_GATE_2F": {"width": 8, "height": 8, "map_hex": "BF", "map_id": 191},
    "SEAFOAM_ISLANDS_1F": {"width": 30, "height": 18, "map_hex": "C0", "map_id": 192},
    "ROUTE_22_GATE": {"width": 10, "height": 8, "map_hex": "C1", "map_id": 193},
    "VICTORY_ROAD_2F": {"width": 30, "height": 18, "map_hex": "C2", "map_id": 194},
    "ROUTE_12_GATE_2F": {"width": 8, "height": 8, "map_hex": "C3", "map_id": 195},
    "VERMILION_TRADE_HOUSE": {"width": 8, "height": 8, "map_hex": "C4", "map_id": 196},
    "DIGLETTS_CAVE": {"width": 40, "height": 36, "map_hex": "C5", "map_id": 197},
    "VICTORY_ROAD_3F": {"width": 30, "height": 18, "map_hex": "C6", "map_id": 198},
    "ROCKET_HIDEOUT_B1F": {"width": 30, "height": 28, "map_hex": "C7", "map_id": 199},
    "ROCKET_HIDEOUT_B2F": {"width": 30, "height": 28, "map_hex": "C8", "map_id": 200},
    "ROCKET_HIDEOUT_B3F": {"width": 30, "height": 28, "map_hex": "C9", "map_id": 201},
    "ROCKET_HIDEOUT_B4F": {"width": 30, "height": 24, "map_hex": "CA", "map_id": 202},
    "ROCKET_HIDEOUT_ELEVATOR": {"width": 6, "height": 8, "map_hex": "CB", "map_id": 203},
    "UNUSED_MAP_CC": {"width": 0, "height": 0, "map_hex": "CC", "map_id": 204},
    "UNUSED_MAP_CD": {"width": 0, "height": 0, "map_hex": "CD", "map_id": 205},
    "UNUSED_MAP_CE": {"width": 0, "height": 0, "map_hex": "CE", "map_id": 206},
    "SILPH_CO_2F": {"width": 30, "height": 18, "map_hex": "CF", "map_id": 207},
    "SILPH_CO_3F": {"width": 30, "height": 18, "map_hex": "D0", "map_id": 208},
    "SILPH_CO_4F": {"width": 30, "height": 18, "map_hex": "D1", "map_id": 209},
    "SILPH_CO_5F": {"width": 30, "height": 18, "map_hex": "D2", "map_id": 210},
    "SILPH_CO_6F": {"width": 26, "height": 18, "map_hex": "D3", "map_id": 211},
    "SILPH_CO_7F": {"width": 26, "height": 18, "map_hex": "D4", "map_id": 212},
    "SILPH_CO_8F": {"width": 26, "height": 18, "map_hex": "D5", "map_id": 213},
    "POKEMON_MANSION_2F": {"width": 30, "height": 28, "map_hex": "D6", "map_id": 214},
    "POKEMON_MANSION_3F": {"width": 30, "height": 18, "map_hex": "D7", "map_id": 215},
    "POKEMON_MANSION_B1F": {"width": 30, "height": 28, "map_hex": "D8", "map_id": 216},
    "SAFARI_ZONE_EAST": {"width": 30, "height": 26, "map_hex": "D9", "map_id": 217},
    "SAFARI_ZONE_NORTH": {"width": 40, "height": 36, "map_hex": "DA", "map_id": 218},
    "SAFARI_ZONE_WEST": {"width": 30, "height": 26, "map_hex": "DB", "map_id": 219},
    "SAFARI_ZONE_CENTER": {"width": 30, "height": 26, "map_hex": "DC", "map_id": 220},
    "SAFARI_ZONE_CENTER_REST_HOUSE": {"width": 8, "height": 8, "map_hex": "DD", "map_id": 221},
    "SAFARI_ZONE_SECRET_HOUSE": {"width": 8, "height": 8, "map_hex": "DE", "map_id": 222},
    "SAFARI_ZONE_WEST_REST_HOUSE": {"width": 8, "height": 8, "map_hex": "DF", "map_id": 223},
    "SAFARI_ZONE_EAST_REST_HOUSE": {"width": 8, "height": 8, "map_hex": "E0", "map_id": 224},
    "SAFARI_ZONE_NORTH_REST_HOUSE": {"width": 8, "height": 8, "map_hex": "E1", "map_id": 225},
    "CERULEAN_CAVE_2F": {"width": 30, "height": 18, "map_hex": "E2", "map_id": 226},
    "CERULEAN_CAVE_B1F": {"width": 30, "height": 18, "map_hex": "E3", "map_id": 227},
    "CERULEAN_CAVE_1F": {"width": 30, "height": 18, "map_hex": "E4", "map_id": 228},
    "NAME_RATERS_HOUSE": {"width": 8, "height": 8, "map_hex": "E5", "map_id": 229},
    "CERULEAN_BADGE_HOUSE": {"width": 8, "height": 8, "map_hex": "E6", "map_id": 230},
    "UNUSED_MAP_E7": {"width": 0, "height": 0, "map_hex": "E7", "map_id": 231},
    "ROCK_TUNNEL_B1F": {"width": 40, "height": 36, "map_hex": "E8", "map_id": 232},
    "SILPH_CO_9F": {"width": 26, "height": 18, "map_hex": "E9", "map_id": 233},
    "SILPH_CO_10F": {"width": 16, "height": 18, "map_hex": "EA", "map_id": 234},
    "SILPH_CO_11F": {"width": 18, "height": 18, "map_hex": "EB", "map_id": 235},
    "SILPH_CO_ELEVATOR": {"width": 4, "height": 4, "map_hex": "EC", "map_id": 236},
    "UNUSED_MAP_ED": {"width": 0, "height": 0, "map_hex": "ED", "map_id": 237},
    "UNUSED_MAP_EE": {"width": 0, "height": 0, "map_hex": "EE", "map_id": 238},
    "TRADE_CENTER": {"width": 10, "height": 8, "map_hex": "EF", "map_id": 239},
    "COLOSSEUM": {"width": 10, "height": 8, "map_hex": "F0", "map_id": 240},
    "UNUSED_MAP_F1": {"width": 0, "height": 0, "map_hex": "F1", "map_id": 241},
    "UNUSED_MAP_F2": {"width": 0, "height": 0, "map_hex": "F2", "map_id": 242},
    "UNUSED_MAP_F3": {"width": 0, "height": 0, "map_hex": "F3", "map_id": 243},
    "UNUSED_MAP_F4": {"width": 0, "height": 0, "map_hex": "F4", "map_id": 244},
    "LORELEIS_ROOM": {"width": 10, "height": 12, "map_hex": "F5", "map_id": 245},
    "BRUNOS_ROOM": {"width": 10, "height": 12, "map_hex": "F6", "map_id": 246},
    "AGATHAS_ROOM": {"width": 10, "height": 12, "map_hex": "F7", "map_id": 247},
    "LAST_MAP": {"width": 0, "height": 0, "map_hex": "FF", "map_id": 255},
}

MAP_ID_REF = {
    0: "PALLET_TOWN",
    1: "VIRIDIAN_CITY",
    2: "PEWTER_CITY",
    3: "CERULEAN_CITY",
    4: "LAVENDER_TOWN",
    5: "VERMILION_CITY",
    6: "CELADON_CITY",
    7: "FUCHSIA_CITY",
    8: "CINNABAR_ISLAND",
    9: "INDIGO_PLATEAU",
    10: "SAFFRON_CITY",
    11: "UNUSED_MAP_0B",
    12: "ROUTE_1",
    13: "ROUTE_2",
    14: "ROUTE_3",
    15: "ROUTE_4",
    16: "ROUTE_5",
    17: "ROUTE_6",
    18: "ROUTE_7",
    19: "ROUTE_8",
    20: "ROUTE_9",
    21: "ROUTE_10",
    22: "ROUTE_11",
    23: "ROUTE_12",
    24: "ROUTE_13",
    25: "ROUTE_14",
    26: "ROUTE_15",
    27: "ROUTE_16",
    28: "ROUTE_17",
    29: "ROUTE_18",
    30: "ROUTE_19",
    31: "ROUTE_20",
    32: "ROUTE_21",
    33: "ROUTE_22",
    34: "ROUTE_23",
    35: "ROUTE_24",
    36: "ROUTE_25",
    37: "REDS_HOUSE_1F",
    38: "REDS_HOUSE_2F",
    39: "BLUES_HOUSE",
    40: "OAKS_LAB",
    41: "VIRIDIAN_POKECENTER",
    42: "VIRIDIAN_MART",
    43: "VIRIDIAN_SCHOOL_HOUSE",
    44: "VIRIDIAN_NICKNAME_HOUSE",
    45: "VIRIDIAN_GYM",
    46: "DIGLETTS_CAVE_ROUTE_2",
    47: "VIRIDIAN_FOREST_NORTH_GATE",
    48: "ROUTE_2_TRADE_HOUSE",
    49: "ROUTE_2_GATE",
    50: "VIRIDIAN_FOREST_SOUTH_GATE",
    51: "VIRIDIAN_FOREST",
    52: "MUSEUM_1F",
    53: "MUSEUM_2F",
    54: "PEWTER_GYM",
    55: "PEWTER_NIDORAN_HOUSE",
    56: "PEWTER_MART",
    57: "PEWTER_SPEECH_HOUSE",
    58: "PEWTER_POKECENTER",
    59: "MT_MOON_1F",
    60: "MT_MOON_B1F",
    61: "MT_MOON_B2F",
    62: "CERULEAN_TRASHED_HOUSE",
    63: "CERULEAN_TRADE_HOUSE",
    64: "CERULEAN_POKECENTER",
    65: "CERULEAN_GYM",
    66: "BIKE_SHOP",
    67: "CERULEAN_MART",
    68: "MT_MOON_POKECENTER",
    69: "CERULEAN_TRASHED_HOUSE_COPY",
    70: "ROUTE_5_GATE",
    71: "UNDERGROUND_PATH_ROUTE_5",
    72: "DAYCARE",
    73: "ROUTE_6_GATE",
    74: "UNDERGROUND_PATH_ROUTE_6",
    75: "UNDERGROUND_PATH_ROUTE_6_COPY",
    76: "ROUTE_7_GATE",
    77: "UNDERGROUND_PATH_ROUTE_7",
    78: "UNDERGROUND_PATH_ROUTE_7_COPY",
    79: "ROUTE_8_GATE",
    80: "UNDERGROUND_PATH_ROUTE_8",
    81: "ROCK_TUNNEL_POKECENTER",
    82: "ROCK_TUNNEL_1F",
    83: "POWER_PLANT",
    84: "ROUTE_11_GATE_1F",
    85: "DIGLETTS_CAVE_ROUTE_11",
    86: "ROUTE_11_GATE_2F",
    87: "ROUTE_12_GATE_1F",
    88: "BILLS_HOUSE",
    89: "VERMILION_POKECENTER",
    90: "POKEMON_FAN_CLUB",
    91: "VERMILION_MART",
    92: "VERMILION_GYM",
    93: "VERMILION_PIDGEY_HOUSE",
    94: "VERMILION_DOCK",
    95: "SS_ANNE_1F",
    96: "SS_ANNE_2F",
    97: "SS_ANNE_3F",
    98: "SS_ANNE_B1F",
    99: "SS_ANNE_BOW",
    100: "SS_ANNE_KITCHEN",
    101: "SS_ANNE_CAPTAINS_ROOM",
    102: "SS_ANNE_1F_ROOMS",
    103: "SS_ANNE_2F_ROOMS",
    104: "SS_ANNE_B1F_ROOMS",
    105: "UNUSED_MAP_69",
    106: "UNUSED_MAP_6A",
    107: "UNUSED_MAP_6B",
    108: "VICTORY_ROAD_1F",
    109: "UNUSED_MAP_6D",
    110: "UNUSED_MAP_6E",
    111: "UNUSED_MAP_6F",
    112: "UNUSED_MAP_70",
    113: "LANCES_ROOM",
    114: "UNUSED_MAP_72",
    115: "UNUSED_MAP_73",
    116: "UNUSED_MAP_74",
    117: "UNUSED_MAP_75",
    118: "HALL_OF_FAME",
    119: "UNDERGROUND_PATH_NORTH_SOUTH",
    120: "CHAMPIONS_ROOM",
    121: "UNDERGROUND_PATH_WEST_EAST",
    122: "CELADON_MART_1F",
    123: "CELADON_MART_2F",
    124: "CELADON_MART_3F",
    125: "CELADON_MART_4F",
    126: "CELADON_MART_ROOF",
    127: "CELADON_MART_ELEVATOR",
    128: "CELADON_MANSION_1F",
    129: "CELADON_MANSION_2F",
    130: "CELADON_MANSION_3F",
    131: "CELADON_MANSION_ROOF",
    132: "CELADON_MANSION_ROOF_HOUSE",
    133: "CELADON_POKECENTER",
    134: "CELADON_GYM",
    135: "GAME_CORNER",
    136: "CELADON_MART_5F",
    137: "GAME_CORNER_PRIZE_ROOM",
    138: "CELADON_DINER",
    139: "CELADON_CHIEF_HOUSE",
    140: "CELADON_HOTEL",
    141: "LAVENDER_POKECENTER",
    142: "POKEMON_TOWER_1F",
    143: "POKEMON_TOWER_2F",
    144: "POKEMON_TOWER_3F",
    145: "POKEMON_TOWER_4F",
    146: "POKEMON_TOWER_5F",
    147: "POKEMON_TOWER_6F",
    148: "POKEMON_TOWER_7F",
    149: "MR_FUJIS_HOUSE",
    150: "LAVENDER_MART",
    151: "LAVENDER_CUBONE_HOUSE",
    152: "FUCHSIA_MART",
    153: "FUCHSIA_BILLS_GRANDPAS_HOUSE",
    154: "FUCHSIA_POKECENTER",
    155: "WARDENS_HOUSE",
    156: "SAFARI_ZONE_GATE",
    157: "FUCHSIA_GYM",
    158: "FUCHSIA_MEETING_ROOM",
    159: "SEAFOAM_ISLANDS_B1F",
    160: "SEAFOAM_ISLANDS_B2F",
    161: "SEAFOAM_ISLANDS_B3F",
    162: "SEAFOAM_ISLANDS_B4F",
    163: "VERMILION_OLD_ROD_HOUSE",
    164: "FUCHSIA_GOOD_ROD_HOUSE",
    165: "POKEMON_MANSION_1F",
    166: "CINNABAR_GYM",
    167: "CINNABAR_LAB",
    168: "CINNABAR_LAB_TRADE_ROOM",
    169: "CINNABAR_LAB_METRONOME_ROOM",
    170: "CINNABAR_LAB_FOSSIL_ROOM",
    171: "CINNABAR_POKECENTER",
    172: "CINNABAR_MART",
    173: "CINNABAR_MART_COPY",
    174: "INDIGO_PLATEAU_LOBBY",
    175: "COPYCATS_HOUSE_1F",
    176: "COPYCATS_HOUSE_2F",
    177: "FIGHTING_DOJO",
    178: "SAFFRON_GYM",
    179: "SAFFRON_PIDGEY_HOUSE",
    180: "SAFFRON_MART",
    181: "SILPH_CO_1F",
    182: "SAFFRON_POKECENTER",
    183: "MR_PSYCHICS_HOUSE",
    184: "ROUTE_15_GATE_1F",
    185: "ROUTE_15_GATE_2F",
    186: "ROUTE_16_GATE_1F",
    187: "ROUTE_16_GATE_2F",
    188: "ROUTE_16_FLY_HOUSE",
    189: "ROUTE_12_SUPER_ROD_HOUSE",
    190: "ROUTE_18_GATE_1F",
    191: "ROUTE_18_GATE_2F",
    192: "SEAFOAM_ISLANDS_1F",
    193: "ROUTE_22_GATE",
    194: "VICTORY_ROAD_2F",
    195: "ROUTE_12_GATE_2F",
    196: "VERMILION_TRADE_HOUSE",
    197: "DIGLETTS_CAVE",
    198: "VICTORY_ROAD_3F",
    199: "ROCKET_HIDEOUT_B1F",
    200: "ROCKET_HIDEOUT_B2F",
    201: "ROCKET_HIDEOUT_B3F",
    202: "ROCKET_HIDEOUT_B4F",
    203: "ROCKET_HIDEOUT_ELEVATOR",
    204: "UNUSED_MAP_CC",
    205: "UNUSED_MAP_CD",
    206: "UNUSED_MAP_CE",
    207: "SILPH_CO_2F",
    208: "SILPH_CO_3F",
    209: "SILPH_CO_4F",
    210: "SILPH_CO_5F",
    211: "SILPH_CO_6F",
    212: "SILPH_CO_7F",
    213: "SILPH_CO_8F",
    214: "POKEMON_MANSION_2F",
    215: "POKEMON_MANSION_3F",
    216: "POKEMON_MANSION_B1F",
    217: "SAFARI_ZONE_EAST",
    218: "SAFARI_ZONE_NORTH",
    219: "SAFARI_ZONE_WEST",
    220: "SAFARI_ZONE_CENTER",
    221: "SAFARI_ZONE_CENTER_REST_HOUSE",
    222: "SAFARI_ZONE_SECRET_HOUSE",
    223: "SAFARI_ZONE_WEST_REST_HOUSE",
    224: "SAFARI_ZONE_EAST_REST_HOUSE",
    225: "SAFARI_ZONE_NORTH_REST_HOUSE",
    226: "CERULEAN_CAVE_2F",
    227: "CERULEAN_CAVE_B1F",
    228: "CERULEAN_CAVE_1F",
    229: "NAME_RATERS_HOUSE",
    230: "CERULEAN_BADGE_HOUSE",
    231: "UNUSED_MAP_E7",
    232: "ROCK_TUNNEL_B1F",
    233: "SILPH_CO_9F",
    234: "SILPH_CO_10F",
    235: "SILPH_CO_11F",
    236: "SILPH_CO_ELEVATOR",
    237: "UNUSED_MAP_ED",
    238: "UNUSED_MAP_EE",
    239: "TRADE_CENTER",
    240: "COLOSSEUM",
    241: "UNUSED_MAP_F1",
    242: "UNUSED_MAP_F2",
    243: "UNUSED_MAP_F3",
    244: "UNUSED_MAP_F4",
    245: "LORELEIS_ROOM",
    246: "BRUNOS_ROOM",
    247: "AGATHAS_ROOM",
    255: "LAST_MAP",
}

WARP_DICT = {
    "AGATHAS_ROOM": [
        {"x": 4, "y": 11, "target_map_name": "BRUNOS_ROOM", "target_map_id": 246, "warp_id": 3},
        {"x": 5, "y": 11, "target_map_name": "BRUNOS_ROOM", "target_map_id": 246, "warp_id": 4},
        {"x": 4, "y": 0, "target_map_name": "LANCES_ROOM", "target_map_id": 113, "warp_id": 1},
        {"x": 5, "y": 0, "target_map_name": "LANCES_ROOM", "target_map_id": 113, "warp_id": 1},
    ],
    "BIKE_SHOP": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "BILLS_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "BLUES_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "BRUNOS_ROOM": [
        {"x": 4, "y": 11, "target_map_name": "LORELEIS_ROOM", "target_map_id": 245, "warp_id": 3},
        {"x": 5, "y": 11, "target_map_name": "LORELEIS_ROOM", "target_map_id": 245, "warp_id": 4},
        {"x": 4, "y": 0, "target_map_name": "AGATHAS_ROOM", "target_map_id": 247, "warp_id": 1},
        {"x": 5, "y": 0, "target_map_name": "AGATHAS_ROOM", "target_map_id": 247, "warp_id": 2},
    ],
    "CELADON_CHIEF_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 12},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 12},
    ],
    "CELADON_CITY": [
        {"x": 8, "y": 13, "target_map_name": "CELADON_MART_1F", "target_map_id": 122, "warp_id": 1},
        {
            "x": 10,
            "y": 13,
            "target_map_name": "CELADON_MART_1F",
            "target_map_id": 122,
            "warp_id": 3,
        },
        {
            "x": 24,
            "y": 9,
            "target_map_name": "CELADON_MANSION_1F",
            "target_map_id": 128,
            "warp_id": 1,
        },
        {
            "x": 24,
            "y": 3,
            "target_map_name": "CELADON_MANSION_1F",
            "target_map_id": 128,
            "warp_id": 3,
        },
        {
            "x": 25,
            "y": 3,
            "target_map_name": "CELADON_MANSION_1F",
            "target_map_id": 128,
            "warp_id": 3,
        },
        {
            "x": 41,
            "y": 9,
            "target_map_name": "CELADON_POKECENTER",
            "target_map_id": 133,
            "warp_id": 1,
        },
        {"x": 12, "y": 27, "target_map_name": "CELADON_GYM", "target_map_id": 134, "warp_id": 1},
        {"x": 28, "y": 19, "target_map_name": "GAME_CORNER", "target_map_id": 135, "warp_id": 1},
        {
            "x": 39,
            "y": 19,
            "target_map_name": "CELADON_MART_5F",
            "target_map_id": 136,
            "warp_id": 1,
        },
        {
            "x": 33,
            "y": 19,
            "target_map_name": "GAME_CORNER_PRIZE_ROOM",
            "target_map_id": 137,
            "warp_id": 1,
        },
        {"x": 31, "y": 27, "target_map_name": "CELADON_DINER", "target_map_id": 138, "warp_id": 1},
        {
            "x": 35,
            "y": 27,
            "target_map_name": "CELADON_CHIEF_HOUSE",
            "target_map_id": 139,
            "warp_id": 1,
        },
        {"x": 43, "y": 27, "target_map_name": "CELADON_HOTEL", "target_map_id": 140, "warp_id": 1},
    ],
    "CELADON_DINER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 11},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 11},
    ],
    "CELADON_GYM": [
        {"x": 4, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 5, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
    ],
    "CELADON_HOTEL": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 13},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 13},
    ],
    "CELADON_MANSION_1F": [
        {"x": 4, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 5, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {
            "x": 7,
            "y": 1,
            "target_map_name": "CELADON_MANSION_2F",
            "target_map_id": 129,
            "warp_id": 2,
        },
        {
            "x": 2,
            "y": 1,
            "target_map_name": "CELADON_MANSION_2F",
            "target_map_id": 129,
            "warp_id": 3,
        },
    ],
    "CELADON_MANSION_2F": [
        {
            "x": 6,
            "y": 1,
            "target_map_name": "CELADON_MANSION_3F",
            "target_map_id": 130,
            "warp_id": 1,
        },
        {
            "x": 7,
            "y": 1,
            "target_map_name": "CELADON_MANSION_1F",
            "target_map_id": 128,
            "warp_id": 4,
        },
        {
            "x": 2,
            "y": 1,
            "target_map_name": "CELADON_MANSION_1F",
            "target_map_id": 128,
            "warp_id": 5,
        },
        {
            "x": 4,
            "y": 1,
            "target_map_name": "CELADON_MANSION_3F",
            "target_map_id": 130,
            "warp_id": 4,
        },
    ],
    "CELADON_MANSION_3F": [
        {
            "x": 6,
            "y": 1,
            "target_map_name": "CELADON_MANSION_2F",
            "target_map_id": 129,
            "warp_id": 1,
        },
        {
            "x": 7,
            "y": 1,
            "target_map_name": "CELADON_MANSION_ROOF",
            "target_map_id": 131,
            "warp_id": 1,
        },
        {
            "x": 2,
            "y": 1,
            "target_map_name": "CELADON_MANSION_ROOF",
            "target_map_id": 131,
            "warp_id": 2,
        },
        {
            "x": 4,
            "y": 1,
            "target_map_name": "CELADON_MANSION_2F",
            "target_map_id": 129,
            "warp_id": 4,
        },
    ],
    "CELADON_MANSION_ROOF": [
        {
            "x": 6,
            "y": 1,
            "target_map_name": "CELADON_MANSION_3F",
            "target_map_id": 130,
            "warp_id": 2,
        },
        {
            "x": 2,
            "y": 1,
            "target_map_name": "CELADON_MANSION_3F",
            "target_map_id": 130,
            "warp_id": 3,
        },
        {
            "x": 2,
            "y": 7,
            "target_map_name": "CELADON_MANSION_ROOF_HOUSE",
            "target_map_id": 132,
            "warp_id": 1,
        },
    ],
    "CELADON_MANSION_ROOF_HOUSE": [
        {
            "x": 2,
            "y": 7,
            "target_map_name": "CELADON_MANSION_ROOF",
            "target_map_id": 131,
            "warp_id": 3,
        },
        {
            "x": 3,
            "y": 7,
            "target_map_name": "CELADON_MANSION_ROOF",
            "target_map_id": 131,
            "warp_id": 3,
        },
    ],
    "CELADON_MART_1F": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 16, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 17, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 12, "y": 1, "target_map_name": "CELADON_MART_2F", "target_map_id": 123, "warp_id": 1},
        {
            "x": 1,
            "y": 1,
            "target_map_name": "CELADON_MART_ELEVATOR",
            "target_map_id": 127,
            "warp_id": 1,
        },
    ],
    "CELADON_MART_2F": [
        {"x": 12, "y": 1, "target_map_name": "CELADON_MART_1F", "target_map_id": 122, "warp_id": 5},
        {"x": 16, "y": 1, "target_map_name": "CELADON_MART_3F", "target_map_id": 124, "warp_id": 2},
        {
            "x": 1,
            "y": 1,
            "target_map_name": "CELADON_MART_ELEVATOR",
            "target_map_id": 127,
            "warp_id": 1,
        },
    ],
    "CELADON_MART_3F": [
        {"x": 12, "y": 1, "target_map_name": "CELADON_MART_4F", "target_map_id": 125, "warp_id": 1},
        {"x": 16, "y": 1, "target_map_name": "CELADON_MART_2F", "target_map_id": 123, "warp_id": 2},
        {
            "x": 1,
            "y": 1,
            "target_map_name": "CELADON_MART_ELEVATOR",
            "target_map_id": 127,
            "warp_id": 1,
        },
    ],
    "CELADON_MART_4F": [
        {"x": 12, "y": 1, "target_map_name": "CELADON_MART_3F", "target_map_id": 124, "warp_id": 1},
        {"x": 16, "y": 1, "target_map_name": "CELADON_MART_5F", "target_map_id": 136, "warp_id": 2},
        {
            "x": 1,
            "y": 1,
            "target_map_name": "CELADON_MART_ELEVATOR",
            "target_map_id": 127,
            "warp_id": 1,
        },
    ],
    "CELADON_MART_5F": [
        {
            "x": 12,
            "y": 1,
            "target_map_name": "CELADON_MART_ROOF",
            "target_map_id": 126,
            "warp_id": 1,
        },
        {"x": 16, "y": 1, "target_map_name": "CELADON_MART_4F", "target_map_id": 125, "warp_id": 2},
        {
            "x": 1,
            "y": 1,
            "target_map_name": "CELADON_MART_ELEVATOR",
            "target_map_id": 127,
            "warp_id": 1,
        },
    ],
    "CELADON_MART_ELEVATOR": [
        {"x": 1, "y": 3, "target_map_name": "CELADON_MART_1F", "target_map_id": 122, "warp_id": 6},
        {"x": 2, "y": 3, "target_map_name": "CELADON_MART_1F", "target_map_id": 122, "warp_id": 6},
    ],
    "CELADON_MART_ROOF": [
        {"x": 15, "y": 2, "target_map_name": "CELADON_MART_5F", "target_map_id": 136, "warp_id": 1}
    ],
    "CELADON_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "CERULEAN_BADGE_HOUSE": [
        {"x": 2, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 10},
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
    ],
    "CERULEAN_CAVE_1F": [
        {"x": 24, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 25, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {
            "x": 27,
            "y": 1,
            "target_map_name": "CERULEAN_CAVE_2F",
            "target_map_id": 226,
            "warp_id": 1,
        },
        {
            "x": 23,
            "y": 7,
            "target_map_name": "CERULEAN_CAVE_2F",
            "target_map_id": 226,
            "warp_id": 2,
        },
        {
            "x": 18,
            "y": 9,
            "target_map_name": "CERULEAN_CAVE_2F",
            "target_map_id": 226,
            "warp_id": 3,
        },
        {"x": 7, "y": 1, "target_map_name": "CERULEAN_CAVE_2F", "target_map_id": 226, "warp_id": 4},
        {"x": 1, "y": 3, "target_map_name": "CERULEAN_CAVE_2F", "target_map_id": 226, "warp_id": 5},
        {
            "x": 3,
            "y": 11,
            "target_map_name": "CERULEAN_CAVE_2F",
            "target_map_id": 226,
            "warp_id": 6,
        },
        {
            "x": 0,
            "y": 6,
            "target_map_name": "CERULEAN_CAVE_B1F",
            "target_map_id": 227,
            "warp_id": 1,
        },
    ],
    "CERULEAN_CAVE_2F": [
        {
            "x": 29,
            "y": 1,
            "target_map_name": "CERULEAN_CAVE_1F",
            "target_map_id": 228,
            "warp_id": 3,
        },
        {
            "x": 22,
            "y": 6,
            "target_map_name": "CERULEAN_CAVE_1F",
            "target_map_id": 228,
            "warp_id": 4,
        },
        {
            "x": 19,
            "y": 7,
            "target_map_name": "CERULEAN_CAVE_1F",
            "target_map_id": 228,
            "warp_id": 5,
        },
        {"x": 9, "y": 1, "target_map_name": "CERULEAN_CAVE_1F", "target_map_id": 228, "warp_id": 6},
        {"x": 1, "y": 3, "target_map_name": "CERULEAN_CAVE_1F", "target_map_id": 228, "warp_id": 7},
        {
            "x": 3,
            "y": 11,
            "target_map_name": "CERULEAN_CAVE_1F",
            "target_map_id": 228,
            "warp_id": 8,
        },
    ],
    "CERULEAN_CAVE_B1F": [
        {"x": 3, "y": 6, "target_map_name": "CERULEAN_CAVE_1F", "target_map_id": 228, "warp_id": 9}
    ],
    "CERULEAN_CITY": [
        {
            "x": 27,
            "y": 11,
            "target_map_name": "CERULEAN_TRASHED_HOUSE",
            "target_map_id": 62,
            "warp_id": 1,
        },
        {
            "x": 13,
            "y": 15,
            "target_map_name": "CERULEAN_TRADE_HOUSE",
            "target_map_id": 63,
            "warp_id": 1,
        },
        {
            "x": 19,
            "y": 17,
            "target_map_name": "CERULEAN_POKECENTER",
            "target_map_id": 64,
            "warp_id": 1,
        },
        {"x": 30, "y": 19, "target_map_name": "CERULEAN_GYM", "target_map_id": 65, "warp_id": 1},
        {"x": 13, "y": 25, "target_map_name": "BIKE_SHOP", "target_map_id": 66, "warp_id": 1},
        {"x": 25, "y": 25, "target_map_name": "CERULEAN_MART", "target_map_id": 67, "warp_id": 1},
        {
            "x": 4,
            "y": 11,
            "target_map_name": "CERULEAN_CAVE_1F",
            "target_map_id": 228,
            "warp_id": 1,
        },
        {
            "x": 27,
            "y": 9,
            "target_map_name": "CERULEAN_TRASHED_HOUSE",
            "target_map_id": 62,
            "warp_id": 3,
        },
        {
            "x": 9,
            "y": 11,
            "target_map_name": "CERULEAN_BADGE_HOUSE",
            "target_map_id": 230,
            "warp_id": 2,
        },
        {
            "x": 9,
            "y": 9,
            "target_map_name": "CERULEAN_BADGE_HOUSE",
            "target_map_id": 230,
            "warp_id": 1,
        },
    ],
    "CERULEAN_GYM": [
        {"x": 4, "y": 13, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 13, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "CERULEAN_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "CERULEAN_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "CERULEAN_TRADE_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "CERULEAN_TRASHED_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
    ],
    "CHAMPIONS_ROOM": [
        {"x": 3, "y": 7, "target_map_name": "LANCES_ROOM", "target_map_id": 113, "warp_id": 2},
        {"x": 4, "y": 7, "target_map_name": "LANCES_ROOM", "target_map_id": 113, "warp_id": 3},
        {"x": 3, "y": 0, "target_map_name": "HALL_OF_FAME", "target_map_id": 118, "warp_id": 1},
        {"x": 4, "y": 0, "target_map_name": "HALL_OF_FAME", "target_map_id": 118, "warp_id": 1},
    ],
    "CINNABAR_GYM": [
        {"x": 16, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 17, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "CINNABAR_ISLAND": [
        {
            "x": 6,
            "y": 3,
            "target_map_name": "POKEMON_MANSION_1F",
            "target_map_id": 165,
            "warp_id": 2,
        },
        {"x": 18, "y": 3, "target_map_name": "CINNABAR_GYM", "target_map_id": 166, "warp_id": 1},
        {"x": 6, "y": 9, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 1},
        {
            "x": 11,
            "y": 11,
            "target_map_name": "CINNABAR_POKECENTER",
            "target_map_id": 171,
            "warp_id": 1,
        },
        {"x": 15, "y": 11, "target_map_name": "CINNABAR_MART", "target_map_id": 172, "warp_id": 1},
    ],
    "CINNABAR_LAB": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {
            "x": 8,
            "y": 4,
            "target_map_name": "CINNABAR_LAB_TRADE_ROOM",
            "target_map_id": 168,
            "warp_id": 1,
        },
        {
            "x": 12,
            "y": 4,
            "target_map_name": "CINNABAR_LAB_METRONOME_ROOM",
            "target_map_id": 169,
            "warp_id": 1,
        },
        {
            "x": 16,
            "y": 4,
            "target_map_name": "CINNABAR_LAB_FOSSIL_ROOM",
            "target_map_id": 170,
            "warp_id": 1,
        },
    ],
    "CINNABAR_LAB_FOSSIL_ROOM": [
        {"x": 2, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 5},
    ],
    "CINNABAR_LAB_METRONOME_ROOM": [
        {"x": 2, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 4},
        {"x": 3, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 4},
    ],
    "CINNABAR_LAB_TRADE_ROOM": [
        {"x": 2, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 3},
        {"x": 3, "y": 7, "target_map_name": "CINNABAR_LAB", "target_map_id": 167, "warp_id": 3},
    ],
    "CINNABAR_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "CINNABAR_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "COLOSSEUM": [],
    "COPYCATS_HOUSE_1F": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {
            "x": 7,
            "y": 1,
            "target_map_name": "COPYCATS_HOUSE_2F",
            "target_map_id": 176,
            "warp_id": 1,
        },
    ],
    "COPYCATS_HOUSE_2F": [
        {"x": 7, "y": 1, "target_map_name": "COPYCATS_HOUSE_1F", "target_map_id": 175, "warp_id": 3}
    ],
    "DAYCARE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "DIGLETTS_CAVE": [
        {
            "x": 5,
            "y": 5,
            "target_map_name": "DIGLETTS_CAVE_ROUTE_2",
            "target_map_id": 46,
            "warp_id": 3,
        },
        {
            "x": 37,
            "y": 31,
            "target_map_name": "DIGLETTS_CAVE_ROUTE_11",
            "target_map_id": 85,
            "warp_id": 3,
        },
    ],
    "DIGLETTS_CAVE_ROUTE_11": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 4, "target_map_name": "DIGLETTS_CAVE", "target_map_id": 197, "warp_id": 2},
    ],
    "DIGLETTS_CAVE_ROUTE_2": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 4, "target_map_name": "DIGLETTS_CAVE", "target_map_id": 197, "warp_id": 1},
    ],
    "FIGHTING_DOJO": [
        {"x": 4, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 5, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "FUCHSIA_BILLS_GRANDPAS_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "FUCHSIA_CITY": [
        {"x": 5, "y": 13, "target_map_name": "FUCHSIA_MART", "target_map_id": 152, "warp_id": 1},
        {
            "x": 11,
            "y": 27,
            "target_map_name": "FUCHSIA_BILLS_GRANDPAS_HOUSE",
            "target_map_id": 153,
            "warp_id": 1,
        },
        {
            "x": 19,
            "y": 27,
            "target_map_name": "FUCHSIA_POKECENTER",
            "target_map_id": 154,
            "warp_id": 1,
        },
        {"x": 27, "y": 27, "target_map_name": "WARDENS_HOUSE", "target_map_id": 155, "warp_id": 1},
        {
            "x": 18,
            "y": 3,
            "target_map_name": "SAFARI_ZONE_GATE",
            "target_map_id": 156,
            "warp_id": 1,
        },
        {"x": 5, "y": 27, "target_map_name": "FUCHSIA_GYM", "target_map_id": 157, "warp_id": 1},
        {
            "x": 22,
            "y": 13,
            "target_map_name": "FUCHSIA_MEETING_ROOM",
            "target_map_id": 158,
            "warp_id": 1,
        },
        {
            "x": 31,
            "y": 27,
            "target_map_name": "FUCHSIA_GOOD_ROD_HOUSE",
            "target_map_id": 164,
            "warp_id": 2,
        },
        {
            "x": 31,
            "y": 24,
            "target_map_name": "FUCHSIA_GOOD_ROD_HOUSE",
            "target_map_id": 164,
            "warp_id": 1,
        },
    ],
    "FUCHSIA_GOOD_ROD_HOUSE": [
        {"x": 2, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
    ],
    "FUCHSIA_GYM": [
        {"x": 4, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 5, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "FUCHSIA_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "FUCHSIA_MEETING_ROOM": [
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
    ],
    "FUCHSIA_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "GAME_CORNER": [
        {"x": 15, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {"x": 16, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {
            "x": 17,
            "y": 4,
            "target_map_name": "ROCKET_HIDEOUT_B1F",
            "target_map_id": 199,
            "warp_id": 2,
        },
    ],
    "GAME_CORNER_PRIZE_ROOM": [
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 10},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 10},
    ],
    "HALL_OF_FAME": [
        {"x": 4, "y": 7, "target_map_name": "CHAMPIONS_ROOM", "target_map_id": 120, "warp_id": 3},
        {"x": 5, "y": 7, "target_map_name": "CHAMPIONS_ROOM", "target_map_id": 120, "warp_id": 4},
    ],
    "INDIGO_PLATEAU": [
        {
            "x": 9,
            "y": 5,
            "target_map_name": "INDIGO_PLATEAU_LOBBY",
            "target_map_id": 174,
            "warp_id": 1,
        },
        {
            "x": 10,
            "y": 5,
            "target_map_name": "INDIGO_PLATEAU_LOBBY",
            "target_map_id": 174,
            "warp_id": 1,
        },
    ],
    "INDIGO_PLATEAU_LOBBY": [
        {"x": 7, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 8, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 8, "y": 0, "target_map_name": "LORELEIS_ROOM", "target_map_id": 245, "warp_id": 1},
    ],
    "LANCES_ROOM": [
        {"x": 24, "y": 16, "target_map_name": "AGATHAS_ROOM", "target_map_id": 247, "warp_id": 3},
        {"x": 5, "y": 0, "target_map_name": "CHAMPIONS_ROOM", "target_map_id": 120, "warp_id": 1},
        {"x": 6, "y": 0, "target_map_name": "CHAMPIONS_ROOM", "target_map_id": 120, "warp_id": 1},
    ],
    "LAVENDER_CUBONE_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "LAVENDER_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "LAVENDER_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "LAVENDER_TOWN": [
        {
            "x": 3,
            "y": 5,
            "target_map_name": "LAVENDER_POKECENTER",
            "target_map_id": 141,
            "warp_id": 1,
        },
        {
            "x": 14,
            "y": 5,
            "target_map_name": "POKEMON_TOWER_1F",
            "target_map_id": 142,
            "warp_id": 1,
        },
        {"x": 7, "y": 9, "target_map_name": "MR_FUJIS_HOUSE", "target_map_id": 149, "warp_id": 1},
        {"x": 15, "y": 13, "target_map_name": "LAVENDER_MART", "target_map_id": 150, "warp_id": 1},
        {
            "x": 3,
            "y": 13,
            "target_map_name": "LAVENDER_CUBONE_HOUSE",
            "target_map_id": 151,
            "warp_id": 1,
        },
        {
            "x": 7,
            "y": 13,
            "target_map_name": "NAME_RATERS_HOUSE",
            "target_map_id": 229,
            "warp_id": 1,
        },
    ],
    "LORELEIS_ROOM": [
        {
            "x": 4,
            "y": 11,
            "target_map_name": "INDIGO_PLATEAU_LOBBY",
            "target_map_id": 174,
            "warp_id": 3,
        },
        {
            "x": 5,
            "y": 11,
            "target_map_name": "INDIGO_PLATEAU_LOBBY",
            "target_map_id": 174,
            "warp_id": 3,
        },
        {"x": 4, "y": 0, "target_map_name": "BRUNOS_ROOM", "target_map_id": 246, "warp_id": 1},
        {"x": 5, "y": 0, "target_map_name": "BRUNOS_ROOM", "target_map_id": 246, "warp_id": 2},
    ],
    "MR_FUJIS_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "MR_PSYCHICS_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
    ],
    "MT_MOON_1F": [
        {"x": 14, "y": 35, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 15, "y": 35, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 5, "y": 5, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 1},
        {"x": 17, "y": 11, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 3},
        {"x": 25, "y": 15, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 4},
    ],
    "MT_MOON_B1F": [
        {"x": 5, "y": 5, "target_map_name": "MT_MOON_1F", "target_map_id": 59, "warp_id": 3},
        {"x": 17, "y": 11, "target_map_name": "MT_MOON_B2F", "target_map_id": 61, "warp_id": 1},
        {"x": 25, "y": 9, "target_map_name": "MT_MOON_1F", "target_map_id": 59, "warp_id": 4},
        {"x": 25, "y": 15, "target_map_name": "MT_MOON_1F", "target_map_id": 59, "warp_id": 5},
        {"x": 21, "y": 17, "target_map_name": "MT_MOON_B2F", "target_map_id": 61, "warp_id": 2},
        {"x": 13, "y": 27, "target_map_name": "MT_MOON_B2F", "target_map_id": 61, "warp_id": 3},
        {"x": 23, "y": 3, "target_map_name": "MT_MOON_B2F", "target_map_id": 61, "warp_id": 4},
        {"x": 27, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "MT_MOON_B2F": [
        {"x": 25, "y": 9, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 2},
        {"x": 21, "y": 17, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 5},
        {"x": 15, "y": 27, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 6},
        {"x": 5, "y": 7, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 7},
    ],
    "MT_MOON_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "MUSEUM_1F": [
        {"x": 10, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 11, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 16, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 17, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 7, "y": 7, "target_map_name": "MUSEUM_2F", "target_map_id": 53, "warp_id": 1},
    ],
    "MUSEUM_2F": [
        {"x": 7, "y": 7, "target_map_name": "MUSEUM_1F", "target_map_id": 52, "warp_id": 5}
    ],
    "NAME_RATERS_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "OAKS_LAB": [
        {"x": 4, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 5, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "PALLET_TOWN": [
        {"x": 5, "y": 5, "target_map_name": "REDS_HOUSE_1F", "target_map_id": 37, "warp_id": 1},
        {"x": 13, "y": 5, "target_map_name": "BLUES_HOUSE", "target_map_id": 39, "warp_id": 1},
        {"x": 12, "y": 11, "target_map_name": "OAKS_LAB", "target_map_id": 40, "warp_id": 2},
    ],
    "PEWTER_CITY": [
        {"x": 14, "y": 7, "target_map_name": "MUSEUM_1F", "target_map_id": 52, "warp_id": 1},
        {"x": 19, "y": 5, "target_map_name": "MUSEUM_1F", "target_map_id": 52, "warp_id": 3},
        {"x": 16, "y": 17, "target_map_name": "PEWTER_GYM", "target_map_id": 54, "warp_id": 1},
        {
            "x": 29,
            "y": 13,
            "target_map_name": "PEWTER_NIDORAN_HOUSE",
            "target_map_id": 55,
            "warp_id": 1,
        },
        {"x": 23, "y": 17, "target_map_name": "PEWTER_MART", "target_map_id": 56, "warp_id": 1},
        {
            "x": 7,
            "y": 29,
            "target_map_name": "PEWTER_SPEECH_HOUSE",
            "target_map_id": 57,
            "warp_id": 1,
        },
        {
            "x": 13,
            "y": 25,
            "target_map_name": "PEWTER_POKECENTER",
            "target_map_id": 58,
            "warp_id": 1,
        },
    ],
    "PEWTER_GYM": [
        {"x": 4, "y": 13, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 5, "y": 13, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "PEWTER_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "PEWTER_NIDORAN_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "PEWTER_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
    ],
    "PEWTER_SPEECH_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "POKEMON_FAN_CLUB": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "POKEMON_MANSION_1F": [
        {"x": 4, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 6, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 7, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {
            "x": 5,
            "y": 10,
            "target_map_name": "POKEMON_MANSION_2F",
            "target_map_id": 214,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 23,
            "target_map_name": "POKEMON_MANSION_B1F",
            "target_map_id": 216,
            "warp_id": 1,
        },
        {"x": 26, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 27, "y": 27, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "POKEMON_MANSION_2F": [
        {
            "x": 5,
            "y": 10,
            "target_map_name": "POKEMON_MANSION_1F",
            "target_map_id": 165,
            "warp_id": 5,
        },
        {
            "x": 7,
            "y": 10,
            "target_map_name": "POKEMON_MANSION_3F",
            "target_map_id": 215,
            "warp_id": 1,
        },
        {
            "x": 25,
            "y": 14,
            "target_map_name": "POKEMON_MANSION_3F",
            "target_map_id": 215,
            "warp_id": 3,
        },
        {
            "x": 6,
            "y": 1,
            "target_map_name": "POKEMON_MANSION_3F",
            "target_map_id": 215,
            "warp_id": 2,
        },
    ],
    "POKEMON_MANSION_3F": [
        {
            "x": 7,
            "y": 10,
            "target_map_name": "POKEMON_MANSION_2F",
            "target_map_id": 214,
            "warp_id": 2,
        },
        {
            "x": 6,
            "y": 1,
            "target_map_name": "POKEMON_MANSION_2F",
            "target_map_id": 214,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 14,
            "target_map_name": "POKEMON_MANSION_2F",
            "target_map_id": 214,
            "warp_id": 3,
        },
    ],
    "POKEMON_MANSION_B1F": [
        {
            "x": 23,
            "y": 22,
            "target_map_name": "POKEMON_MANSION_1F",
            "target_map_id": 165,
            "warp_id": 6,
        }
    ],
    "POKEMON_TOWER_1F": [
        {"x": 10, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 11, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_2F",
            "target_map_id": 143,
            "warp_id": 2,
        },
    ],
    "POKEMON_TOWER_2F": [
        {"x": 3, "y": 9, "target_map_name": "POKEMON_TOWER_3F", "target_map_id": 144, "warp_id": 1},
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_1F",
            "target_map_id": 142,
            "warp_id": 3,
        },
    ],
    "POKEMON_TOWER_3F": [
        {"x": 3, "y": 9, "target_map_name": "POKEMON_TOWER_2F", "target_map_id": 143, "warp_id": 1},
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_4F",
            "target_map_id": 145,
            "warp_id": 2,
        },
    ],
    "POKEMON_TOWER_4F": [
        {"x": 3, "y": 9, "target_map_name": "POKEMON_TOWER_5F", "target_map_id": 146, "warp_id": 1},
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_3F",
            "target_map_id": 144,
            "warp_id": 2,
        },
    ],
    "POKEMON_TOWER_5F": [
        {"x": 3, "y": 9, "target_map_name": "POKEMON_TOWER_4F", "target_map_id": 145, "warp_id": 1},
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_6F",
            "target_map_id": 147,
            "warp_id": 1,
        },
    ],
    "POKEMON_TOWER_6F": [
        {
            "x": 18,
            "y": 9,
            "target_map_name": "POKEMON_TOWER_5F",
            "target_map_id": 146,
            "warp_id": 2,
        },
        {
            "x": 9,
            "y": 16,
            "target_map_name": "POKEMON_TOWER_7F",
            "target_map_id": 148,
            "warp_id": 1,
        },
    ],
    "POKEMON_TOWER_7F": [
        {"x": 9, "y": 16, "target_map_name": "POKEMON_TOWER_6F", "target_map_id": 147, "warp_id": 2}
    ],
    "POWER_PLANT": [
        {"x": 4, "y": 35, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 35, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 0, "y": 11, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "REDS_HOUSE_1F": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 7, "y": 1, "target_map_name": "REDS_HOUSE_2F", "target_map_id": 38, "warp_id": 1},
    ],
    "REDS_HOUSE_2F": [
        {"x": 7, "y": 1, "target_map_name": "REDS_HOUSE_1F", "target_map_id": 37, "warp_id": 3}
    ],
    "ROCKET_HIDEOUT_B1F": [
        {
            "x": 23,
            "y": 2,
            "target_map_name": "ROCKET_HIDEOUT_B2F",
            "target_map_id": 200,
            "warp_id": 1,
        },
        {"x": 21, "y": 2, "target_map_name": "GAME_CORNER", "target_map_id": 135, "warp_id": 3},
        {
            "x": 24,
            "y": 19,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 24,
            "target_map_name": "ROCKET_HIDEOUT_B2F",
            "target_map_id": 200,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 19,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 2,
        },
    ],
    "ROCKET_HIDEOUT_B2F": [
        {
            "x": 27,
            "y": 8,
            "target_map_name": "ROCKET_HIDEOUT_B1F",
            "target_map_id": 199,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 8,
            "target_map_name": "ROCKET_HIDEOUT_B3F",
            "target_map_id": 201,
            "warp_id": 1,
        },
        {
            "x": 24,
            "y": 19,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 22,
            "target_map_name": "ROCKET_HIDEOUT_B1F",
            "target_map_id": 199,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 19,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 2,
        },
    ],
    "ROCKET_HIDEOUT_B3F": [
        {
            "x": 25,
            "y": 6,
            "target_map_name": "ROCKET_HIDEOUT_B2F",
            "target_map_id": 200,
            "warp_id": 2,
        },
        {
            "x": 19,
            "y": 18,
            "target_map_name": "ROCKET_HIDEOUT_B4F",
            "target_map_id": 202,
            "warp_id": 1,
        },
    ],
    "ROCKET_HIDEOUT_B4F": [
        {
            "x": 19,
            "y": 10,
            "target_map_name": "ROCKET_HIDEOUT_B3F",
            "target_map_id": 201,
            "warp_id": 2,
        },
        {
            "x": 24,
            "y": 15,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 1,
        },
        {
            "x": 25,
            "y": 15,
            "target_map_name": "ROCKET_HIDEOUT_ELEVATOR",
            "target_map_id": 203,
            "warp_id": 2,
        },
    ],
    "ROCKET_HIDEOUT_ELEVATOR": [
        {
            "x": 2,
            "y": 1,
            "target_map_name": "ROCKET_HIDEOUT_B1F",
            "target_map_id": 199,
            "warp_id": 3,
        },
        {
            "x": 3,
            "y": 1,
            "target_map_name": "ROCKET_HIDEOUT_B1F",
            "target_map_id": 199,
            "warp_id": 5,
        },
    ],
    "ROCK_TUNNEL_1F": [
        {"x": 15, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 15, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 15, "y": 33, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 15, "y": 35, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 37, "y": 3, "target_map_name": "ROCK_TUNNEL_B1F", "target_map_id": 232, "warp_id": 1},
        {"x": 5, "y": 3, "target_map_name": "ROCK_TUNNEL_B1F", "target_map_id": 232, "warp_id": 2},
        {
            "x": 17,
            "y": 11,
            "target_map_name": "ROCK_TUNNEL_B1F",
            "target_map_id": 232,
            "warp_id": 3,
        },
        {
            "x": 37,
            "y": 17,
            "target_map_name": "ROCK_TUNNEL_B1F",
            "target_map_id": 232,
            "warp_id": 4,
        },
    ],
    "ROCK_TUNNEL_B1F": [
        {"x": 33, "y": 25, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 5},
        {"x": 27, "y": 3, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 6},
        {"x": 23, "y": 11, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 7},
        {"x": 3, "y": 3, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 8},
    ],
    "ROCK_TUNNEL_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "ROUTE_1": [],
    "ROUTE_10": [
        {
            "x": 11,
            "y": 19,
            "target_map_name": "ROCK_TUNNEL_POKECENTER",
            "target_map_id": 81,
            "warp_id": 1,
        },
        {"x": 8, "y": 17, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 1},
        {"x": 8, "y": 53, "target_map_name": "ROCK_TUNNEL_1F", "target_map_id": 82, "warp_id": 3},
        {"x": 6, "y": 39, "target_map_name": "POWER_PLANT", "target_map_id": 83, "warp_id": 1},
    ],
    "ROUTE_11": [
        {"x": 49, "y": 8, "target_map_name": "ROUTE_11_GATE_1F", "target_map_id": 84, "warp_id": 1},
        {"x": 49, "y": 9, "target_map_name": "ROUTE_11_GATE_1F", "target_map_id": 84, "warp_id": 2},
        {"x": 58, "y": 8, "target_map_name": "ROUTE_11_GATE_1F", "target_map_id": 84, "warp_id": 3},
        {"x": 58, "y": 9, "target_map_name": "ROUTE_11_GATE_1F", "target_map_id": 84, "warp_id": 4},
        {
            "x": 4,
            "y": 5,
            "target_map_name": "DIGLETTS_CAVE_ROUTE_11",
            "target_map_id": 85,
            "warp_id": 1,
        },
    ],
    "ROUTE_11_GATE_1F": [
        {"x": 0, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 0, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 7, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 7, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 6, "y": 8, "target_map_name": "ROUTE_11_GATE_2F", "target_map_id": 86, "warp_id": 1},
    ],
    "ROUTE_11_GATE_2F": [
        {"x": 7, "y": 7, "target_map_name": "ROUTE_11_GATE_1F", "target_map_id": 84, "warp_id": 5}
    ],
    "ROUTE_12": [
        {
            "x": 10,
            "y": 15,
            "target_map_name": "ROUTE_12_GATE_1F",
            "target_map_id": 87,
            "warp_id": 1,
        },
        {
            "x": 11,
            "y": 15,
            "target_map_name": "ROUTE_12_GATE_1F",
            "target_map_id": 87,
            "warp_id": 2,
        },
        {
            "x": 10,
            "y": 21,
            "target_map_name": "ROUTE_12_GATE_1F",
            "target_map_id": 87,
            "warp_id": 3,
        },
        {
            "x": 11,
            "y": 77,
            "target_map_name": "ROUTE_12_SUPER_ROD_HOUSE",
            "target_map_id": 189,
            "warp_id": 1,
        },
    ],
    "ROUTE_12_GATE_1F": [
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 8, "y": 6, "target_map_name": "ROUTE_12_GATE_2F", "target_map_id": 195, "warp_id": 1},
    ],
    "ROUTE_12_GATE_2F": [
        {"x": 7, "y": 7, "target_map_name": "ROUTE_12_GATE_1F", "target_map_id": 87, "warp_id": 5}
    ],
    "ROUTE_12_SUPER_ROD_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "ROUTE_13": [],
    "ROUTE_14": [],
    "ROUTE_15": [
        {"x": 7, "y": 8, "target_map_name": "ROUTE_15_GATE_1F", "target_map_id": 184, "warp_id": 1},
        {"x": 7, "y": 9, "target_map_name": "ROUTE_15_GATE_1F", "target_map_id": 184, "warp_id": 2},
        {
            "x": 14,
            "y": 8,
            "target_map_name": "ROUTE_15_GATE_1F",
            "target_map_id": 184,
            "warp_id": 3,
        },
        {
            "x": 14,
            "y": 9,
            "target_map_name": "ROUTE_15_GATE_1F",
            "target_map_id": 184,
            "warp_id": 4,
        },
    ],
    "ROUTE_15_GATE_1F": [
        {"x": 0, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 0, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 7, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 7, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 6, "y": 8, "target_map_name": "ROUTE_15_GATE_2F", "target_map_id": 185, "warp_id": 1},
    ],
    "ROUTE_15_GATE_2F": [
        {"x": 7, "y": 7, "target_map_name": "ROUTE_15_GATE_1F", "target_map_id": 184, "warp_id": 5}
    ],
    "ROUTE_16": [
        {
            "x": 17,
            "y": 10,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 1,
        },
        {
            "x": 17,
            "y": 11,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 2,
        },
        {
            "x": 24,
            "y": 10,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 3,
        },
        {
            "x": 24,
            "y": 11,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 4,
        },
        {
            "x": 17,
            "y": 4,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 5,
        },
        {
            "x": 17,
            "y": 5,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 6,
        },
        {
            "x": 24,
            "y": 4,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 7,
        },
        {
            "x": 24,
            "y": 5,
            "target_map_name": "ROUTE_16_GATE_1F",
            "target_map_id": 186,
            "warp_id": 8,
        },
        {
            "x": 7,
            "y": 5,
            "target_map_name": "ROUTE_16_FLY_HOUSE",
            "target_map_id": 188,
            "warp_id": 1,
        },
    ],
    "ROUTE_16_FLY_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
    ],
    "ROUTE_16_GATE_1F": [
        {"x": 0, "y": 8, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 0, "y": 9, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 7, "y": 8, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 7, "y": 9, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 0, "y": 2, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 0, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 7, "y": 2, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 7, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {
            "x": 6,
            "y": 12,
            "target_map_name": "ROUTE_16_GATE_2F",
            "target_map_id": 187,
            "warp_id": 1,
        },
    ],
    "ROUTE_16_GATE_2F": [
        {"x": 7, "y": 7, "target_map_name": "ROUTE_16_GATE_1F", "target_map_id": 186, "warp_id": 9}
    ],
    "ROUTE_17": [],
    "ROUTE_18": [
        {
            "x": 33,
            "y": 8,
            "target_map_name": "ROUTE_18_GATE_1F",
            "target_map_id": 190,
            "warp_id": 1,
        },
        {
            "x": 33,
            "y": 9,
            "target_map_name": "ROUTE_18_GATE_1F",
            "target_map_id": 190,
            "warp_id": 2,
        },
        {
            "x": 40,
            "y": 8,
            "target_map_name": "ROUTE_18_GATE_1F",
            "target_map_id": 190,
            "warp_id": 3,
        },
        {
            "x": 40,
            "y": 9,
            "target_map_name": "ROUTE_18_GATE_1F",
            "target_map_id": 190,
            "warp_id": 4,
        },
    ],
    "ROUTE_18_GATE_1F": [
        {"x": 0, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 0, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 7, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 7, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 6, "y": 8, "target_map_name": "ROUTE_18_GATE_2F", "target_map_id": 191, "warp_id": 1},
    ],
    "ROUTE_18_GATE_2F": [
        {"x": 7, "y": 7, "target_map_name": "ROUTE_18_GATE_1F", "target_map_id": 190, "warp_id": 5}
    ],
    "ROUTE_19": [],
    "ROUTE_2": [
        {
            "x": 12,
            "y": 9,
            "target_map_name": "DIGLETTS_CAVE_ROUTE_2",
            "target_map_id": 46,
            "warp_id": 1,
        },
        {
            "x": 3,
            "y": 11,
            "target_map_name": "VIRIDIAN_FOREST_NORTH_GATE",
            "target_map_id": 47,
            "warp_id": 2,
        },
        {
            "x": 15,
            "y": 19,
            "target_map_name": "ROUTE_2_TRADE_HOUSE",
            "target_map_id": 48,
            "warp_id": 1,
        },
        {"x": 16, "y": 35, "target_map_name": "ROUTE_2_GATE", "target_map_id": 49, "warp_id": 2},
        {"x": 15, "y": 39, "target_map_name": "ROUTE_2_GATE", "target_map_id": 49, "warp_id": 3},
        {
            "x": 3,
            "y": 43,
            "target_map_name": "VIRIDIAN_FOREST_SOUTH_GATE",
            "target_map_id": 50,
            "warp_id": 3,
        },
    ],
    "ROUTE_20": [
        {
            "x": 48,
            "y": 5,
            "target_map_name": "SEAFOAM_ISLANDS_1F",
            "target_map_id": 192,
            "warp_id": 1,
        },
        {
            "x": 58,
            "y": 9,
            "target_map_name": "SEAFOAM_ISLANDS_1F",
            "target_map_id": 192,
            "warp_id": 3,
        },
    ],
    "ROUTE_21": [],
    "ROUTE_22": [
        {"x": 8, "y": 5, "target_map_name": "ROUTE_22_GATE", "target_map_id": 193, "warp_id": 1}
    ],
    "ROUTE_22_GATE": [
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "ROUTE_23": [
        {"x": 7, "y": 139, "target_map_name": "ROUTE_22_GATE", "target_map_id": 193, "warp_id": 3},
        {"x": 8, "y": 139, "target_map_name": "ROUTE_22_GATE", "target_map_id": 193, "warp_id": 4},
        {"x": 4, "y": 31, "target_map_name": "VICTORY_ROAD_1F", "target_map_id": 108, "warp_id": 1},
        {
            "x": 14,
            "y": 31,
            "target_map_name": "VICTORY_ROAD_2F",
            "target_map_id": 194,
            "warp_id": 2,
        },
    ],
    "ROUTE_24": [],
    "ROUTE_25": [
        {"x": 45, "y": 3, "target_map_name": "BILLS_HOUSE", "target_map_id": 88, "warp_id": 1}
    ],
    "ROUTE_2_GATE": [
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "ROUTE_2_TRADE_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "ROUTE_3": [],
    "ROUTE_4": [
        {
            "x": 11,
            "y": 5,
            "target_map_name": "MT_MOON_POKECENTER",
            "target_map_id": 68,
            "warp_id": 1,
        },
        {"x": 18, "y": 5, "target_map_name": "MT_MOON_1F", "target_map_id": 59, "warp_id": 1},
        {"x": 24, "y": 5, "target_map_name": "MT_MOON_B1F", "target_map_id": 60, "warp_id": 8},
    ],
    "ROUTE_5": [
        {"x": 10, "y": 29, "target_map_name": "ROUTE_5_GATE", "target_map_id": 70, "warp_id": 4},
        {"x": 9, "y": 29, "target_map_name": "ROUTE_5_GATE", "target_map_id": 70, "warp_id": 3},
        {"x": 10, "y": 33, "target_map_name": "ROUTE_5_GATE", "target_map_id": 70, "warp_id": 1},
        {
            "x": 17,
            "y": 27,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_5",
            "target_map_id": 71,
            "warp_id": 1,
        },
        {"x": 10, "y": 21, "target_map_name": "DAYCARE", "target_map_id": 72, "warp_id": 1},
    ],
    "ROUTE_5_GATE": [
        {"x": 3, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "ROUTE_6": [
        {"x": 9, "y": 1, "target_map_name": "ROUTE_6_GATE", "target_map_id": 73, "warp_id": 3},
        {"x": 10, "y": 1, "target_map_name": "ROUTE_6_GATE", "target_map_id": 73, "warp_id": 3},
        {"x": 10, "y": 7, "target_map_name": "ROUTE_6_GATE", "target_map_id": 73, "warp_id": 1},
        {
            "x": 17,
            "y": 13,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_6",
            "target_map_id": 74,
            "warp_id": 1,
        },
    ],
    "ROUTE_6_GATE": [
        {"x": 3, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "ROUTE_7": [
        {"x": 18, "y": 9, "target_map_name": "ROUTE_7_GATE", "target_map_id": 76, "warp_id": 3},
        {"x": 18, "y": 10, "target_map_name": "ROUTE_7_GATE", "target_map_id": 76, "warp_id": 4},
        {"x": 11, "y": 9, "target_map_name": "ROUTE_7_GATE", "target_map_id": 76, "warp_id": 1},
        {"x": 11, "y": 10, "target_map_name": "ROUTE_7_GATE", "target_map_id": 76, "warp_id": 2},
        {
            "x": 5,
            "y": 13,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_7",
            "target_map_id": 77,
            "warp_id": 1,
        },
    ],
    "ROUTE_7_GATE": [
        {"x": 0, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 0, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "ROUTE_8": [
        {"x": 1, "y": 9, "target_map_name": "ROUTE_8_GATE", "target_map_id": 79, "warp_id": 1},
        {"x": 1, "y": 10, "target_map_name": "ROUTE_8_GATE", "target_map_id": 79, "warp_id": 2},
        {"x": 8, "y": 9, "target_map_name": "ROUTE_8_GATE", "target_map_id": 79, "warp_id": 3},
        {"x": 8, "y": 10, "target_map_name": "ROUTE_8_GATE", "target_map_id": 79, "warp_id": 4},
        {
            "x": 13,
            "y": 3,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_8",
            "target_map_id": 80,
            "warp_id": 1,
        },
    ],
    "ROUTE_8_GATE": [
        {"x": 0, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 0, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 5, "y": 3, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 5, "y": 4, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "ROUTE_9": [],
    "SAFARI_ZONE_CENTER": [
        {
            "x": 14,
            "y": 25,
            "target_map_name": "SAFARI_ZONE_GATE",
            "target_map_id": 156,
            "warp_id": 3,
        },
        {
            "x": 15,
            "y": 25,
            "target_map_name": "SAFARI_ZONE_GATE",
            "target_map_id": 156,
            "warp_id": 4,
        },
        {
            "x": 0,
            "y": 10,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 5,
        },
        {
            "x": 0,
            "y": 11,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 6,
        },
        {
            "x": 14,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 5,
        },
        {
            "x": 15,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 6,
        },
        {
            "x": 29,
            "y": 10,
            "target_map_name": "SAFARI_ZONE_EAST",
            "target_map_id": 217,
            "warp_id": 3,
        },
        {
            "x": 29,
            "y": 11,
            "target_map_name": "SAFARI_ZONE_EAST",
            "target_map_id": 217,
            "warp_id": 4,
        },
        {
            "x": 17,
            "y": 19,
            "target_map_name": "SAFARI_ZONE_CENTER_REST_HOUSE",
            "target_map_id": 221,
            "warp_id": 1,
        },
    ],
    "SAFARI_ZONE_CENTER_REST_HOUSE": [
        {
            "x": 2,
            "y": 7,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 9,
        },
        {
            "x": 3,
            "y": 7,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 9,
        },
    ],
    "SAFARI_ZONE_EAST": [
        {
            "x": 0,
            "y": 4,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 7,
        },
        {
            "x": 0,
            "y": 5,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 8,
        },
        {
            "x": 0,
            "y": 22,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 7,
        },
        {
            "x": 0,
            "y": 23,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 7,
        },
        {
            "x": 25,
            "y": 9,
            "target_map_name": "SAFARI_ZONE_EAST_REST_HOUSE",
            "target_map_id": 224,
            "warp_id": 1,
        },
    ],
    "SAFARI_ZONE_EAST_REST_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "SAFARI_ZONE_EAST", "target_map_id": 217, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "SAFARI_ZONE_EAST", "target_map_id": 217, "warp_id": 5},
    ],
    "SAFARI_ZONE_GATE": [
        {"x": 3, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {
            "x": 3,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 1,
        },
        {
            "x": 4,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 2,
        },
    ],
    "SAFARI_ZONE_NORTH": [
        {
            "x": 2,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 1,
        },
        {
            "x": 3,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 2,
        },
        {
            "x": 8,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 3,
        },
        {
            "x": 9,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_WEST",
            "target_map_id": 219,
            "warp_id": 4,
        },
        {
            "x": 20,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 5,
        },
        {
            "x": 21,
            "y": 35,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 6,
        },
        {
            "x": 39,
            "y": 30,
            "target_map_name": "SAFARI_ZONE_EAST",
            "target_map_id": 217,
            "warp_id": 1,
        },
        {
            "x": 39,
            "y": 31,
            "target_map_name": "SAFARI_ZONE_EAST",
            "target_map_id": 217,
            "warp_id": 2,
        },
        {
            "x": 35,
            "y": 3,
            "target_map_name": "SAFARI_ZONE_NORTH_REST_HOUSE",
            "target_map_id": 225,
            "warp_id": 1,
        },
    ],
    "SAFARI_ZONE_NORTH_REST_HOUSE": [
        {
            "x": 2,
            "y": 7,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 9,
        },
        {
            "x": 3,
            "y": 7,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 9,
        },
    ],
    "SAFARI_ZONE_SECRET_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "SAFARI_ZONE_WEST", "target_map_id": 219, "warp_id": 7},
        {"x": 3, "y": 7, "target_map_name": "SAFARI_ZONE_WEST", "target_map_id": 219, "warp_id": 7},
    ],
    "SAFARI_ZONE_WEST": [
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 2,
        },
        {
            "x": 26,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 3,
        },
        {
            "x": 27,
            "y": 0,
            "target_map_name": "SAFARI_ZONE_NORTH",
            "target_map_id": 218,
            "warp_id": 4,
        },
        {
            "x": 29,
            "y": 22,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 3,
        },
        {
            "x": 29,
            "y": 23,
            "target_map_name": "SAFARI_ZONE_CENTER",
            "target_map_id": 220,
            "warp_id": 4,
        },
        {
            "x": 3,
            "y": 3,
            "target_map_name": "SAFARI_ZONE_SECRET_HOUSE",
            "target_map_id": 222,
            "warp_id": 1,
        },
        {
            "x": 11,
            "y": 11,
            "target_map_name": "SAFARI_ZONE_WEST_REST_HOUSE",
            "target_map_id": 223,
            "warp_id": 1,
        },
    ],
    "SAFARI_ZONE_WEST_REST_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "SAFARI_ZONE_WEST", "target_map_id": 219, "warp_id": 8},
        {"x": 3, "y": 7, "target_map_name": "SAFARI_ZONE_WEST", "target_map_id": 219, "warp_id": 8},
    ],
    "SAFFRON_CITY": [
        {
            "x": 7,
            "y": 5,
            "target_map_name": "COPYCATS_HOUSE_1F",
            "target_map_id": 175,
            "warp_id": 1,
        },
        {"x": 26, "y": 3, "target_map_name": "FIGHTING_DOJO", "target_map_id": 177, "warp_id": 1},
        {"x": 34, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 1},
        {
            "x": 13,
            "y": 11,
            "target_map_name": "SAFFRON_PIDGEY_HOUSE",
            "target_map_id": 179,
            "warp_id": 1,
        },
        {"x": 25, "y": 11, "target_map_name": "SAFFRON_MART", "target_map_id": 180, "warp_id": 1},
        {"x": 18, "y": 21, "target_map_name": "SILPH_CO_1F", "target_map_id": 181, "warp_id": 1},
        {
            "x": 9,
            "y": 29,
            "target_map_name": "SAFFRON_POKECENTER",
            "target_map_id": 182,
            "warp_id": 1,
        },
        {
            "x": 29,
            "y": 29,
            "target_map_name": "MR_PSYCHICS_HOUSE",
            "target_map_id": 183,
            "warp_id": 1,
        },
    ],
    "SAFFRON_GYM": [
        {"x": 8, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 9, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 1, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 23},
        {"x": 5, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 16},
        {"x": 1, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 19},
        {"x": 5, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 9},
        {"x": 1, "y": 9, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 28},
        {"x": 5, "y": 9, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 17},
        {"x": 1, "y": 11, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 6},
        {"x": 5, "y": 11, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 14},
        {"x": 1, "y": 15, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 24},
        {"x": 5, "y": 15, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 31},
        {"x": 1, "y": 17, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 18},
        {"x": 5, "y": 17, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 10},
        {"x": 9, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 27},
        {"x": 11, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 4},
        {"x": 9, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 8},
        {"x": 11, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 13},
        {"x": 11, "y": 11, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 5},
        {"x": 11, "y": 15, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 32},
        {"x": 15, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 25},
        {"x": 19, "y": 3, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 29},
        {"x": 15, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 3},
        {"x": 19, "y": 5, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 11},
        {"x": 15, "y": 9, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 21},
        {"x": 19, "y": 9, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 30},
        {"x": 15, "y": 11, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 15},
        {"x": 19, "y": 11, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 7},
        {"x": 15, "y": 15, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 22},
        {"x": 19, "y": 15, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 26},
        {"x": 15, "y": 17, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 12},
        {"x": 19, "y": 17, "target_map_name": "SAFFRON_GYM", "target_map_id": 178, "warp_id": 20},
    ],
    "SAFFRON_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "SAFFRON_PIDGEY_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "SAFFRON_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 7},
    ],
    "SEAFOAM_ISLANDS_1F": [
        {"x": 4, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 5, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 26, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 27, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {
            "x": 7,
            "y": 5,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 2,
        },
        {
            "x": 25,
            "y": 3,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 7,
        },
        {
            "x": 23,
            "y": 15,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 5,
        },
    ],
    "SEAFOAM_ISLANDS_B1F": [
        {
            "x": 4,
            "y": 2,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 1,
        },
        {
            "x": 7,
            "y": 5,
            "target_map_name": "SEAFOAM_ISLANDS_1F",
            "target_map_id": 192,
            "warp_id": 5,
        },
        {
            "x": 13,
            "y": 7,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 3,
        },
        {
            "x": 19,
            "y": 15,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 4,
        },
        {
            "x": 23,
            "y": 15,
            "target_map_name": "SEAFOAM_ISLANDS_1F",
            "target_map_id": 192,
            "warp_id": 7,
        },
        {
            "x": 25,
            "y": 11,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 6,
        },
        {
            "x": 25,
            "y": 3,
            "target_map_name": "SEAFOAM_ISLANDS_1F",
            "target_map_id": 192,
            "warp_id": 6,
        },
    ],
    "SEAFOAM_ISLANDS_B2F": [
        {
            "x": 5,
            "y": 3,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 1,
        },
        {
            "x": 5,
            "y": 13,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 1,
        },
        {
            "x": 13,
            "y": 7,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 3,
        },
        {
            "x": 19,
            "y": 15,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 3,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 11,
            "target_map_name": "SEAFOAM_ISLANDS_B1F",
            "target_map_id": 159,
            "warp_id": 6,
        },
        {
            "x": 25,
            "y": 14,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 5,
        },
    ],
    "SEAFOAM_ISLANDS_B3F": [
        {
            "x": 5,
            "y": 12,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 2,
        },
        {
            "x": 8,
            "y": 6,
            "target_map_name": "SEAFOAM_ISLANDS_B4F",
            "target_map_id": 162,
            "warp_id": 3,
        },
        {
            "x": 25,
            "y": 4,
            "target_map_name": "SEAFOAM_ISLANDS_B4F",
            "target_map_id": 162,
            "warp_id": 4,
        },
        {
            "x": 25,
            "y": 3,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 5,
        },
        {
            "x": 25,
            "y": 14,
            "target_map_name": "SEAFOAM_ISLANDS_B2F",
            "target_map_id": 160,
            "warp_id": 7,
        },
        {
            "x": 20,
            "y": 17,
            "target_map_name": "SEAFOAM_ISLANDS_B4F",
            "target_map_id": 162,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 17,
            "target_map_name": "SEAFOAM_ISLANDS_B4F",
            "target_map_id": 162,
            "warp_id": 2,
        },
    ],
    "SEAFOAM_ISLANDS_B4F": [
        {
            "x": 20,
            "y": 17,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 6,
        },
        {
            "x": 21,
            "y": 17,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 7,
        },
        {
            "x": 11,
            "y": 7,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 2,
        },
        {
            "x": 25,
            "y": 4,
            "target_map_name": "SEAFOAM_ISLANDS_B3F",
            "target_map_id": 161,
            "warp_id": 3,
        },
    ],
    "SILPH_CO_10F": [
        {"x": 8, "y": 0, "target_map_name": "SILPH_CO_9F", "target_map_id": 233, "warp_id": 1},
        {"x": 10, "y": 0, "target_map_name": "SILPH_CO_11F", "target_map_id": 235, "warp_id": 1},
        {
            "x": 12,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 9, "y": 11, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 4},
        {"x": 13, "y": 15, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 6},
        {"x": 13, "y": 7, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 7},
    ],
    "SILPH_CO_11F": [
        {"x": 9, "y": 0, "target_map_name": "SILPH_CO_10F", "target_map_id": 234, "warp_id": 2},
        {
            "x": 13,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 5, "y": 5, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 10},
        {"x": 3, "y": 2, "target_map_name": "SILPH_CO_7F", "target_map_id": 212, "warp_id": 4},
    ],
    "SILPH_CO_1F": [
        {"x": 10, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 11, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 26, "y": 0, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 1},
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 16, "y": 10, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 7},
    ],
    "SILPH_CO_2F": [
        {"x": 24, "y": 0, "target_map_name": "SILPH_CO_1F", "target_map_id": 181, "warp_id": 3},
        {"x": 26, "y": 0, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 1},
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 3, "y": 3, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 7},
        {"x": 13, "y": 3, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 5},
        {"x": 27, "y": 15, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 6},
        {"x": 9, "y": 15, "target_map_name": "SILPH_CO_6F", "target_map_id": 211, "warp_id": 5},
    ],
    "SILPH_CO_3F": [
        {"x": 26, "y": 0, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 2},
        {"x": 24, "y": 0, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 1},
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 23, "y": 11, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 10},
        {"x": 3, "y": 3, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 6},
        {"x": 3, "y": 15, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 7},
        {"x": 27, "y": 3, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 4},
        {"x": 3, "y": 11, "target_map_name": "SILPH_CO_9F", "target_map_id": 233, "warp_id": 4},
        {"x": 11, "y": 11, "target_map_name": "SILPH_CO_7F", "target_map_id": 212, "warp_id": 5},
        {"x": 27, "y": 15, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 4},
    ],
    "SILPH_CO_4F": [
        {"x": 24, "y": 0, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 2},
        {"x": 26, "y": 0, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 2},
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 11, "y": 7, "target_map_name": "SILPH_CO_10F", "target_map_id": 234, "warp_id": 4},
        {"x": 17, "y": 3, "target_map_name": "SILPH_CO_6F", "target_map_id": 211, "warp_id": 4},
        {"x": 3, "y": 15, "target_map_name": "SILPH_CO_10F", "target_map_id": 234, "warp_id": 5},
        {"x": 17, "y": 11, "target_map_name": "SILPH_CO_10F", "target_map_id": 234, "warp_id": 6},
    ],
    "SILPH_CO_5F": [
        {"x": 24, "y": 0, "target_map_name": "SILPH_CO_6F", "target_map_id": 211, "warp_id": 2},
        {"x": 26, "y": 0, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 2},
        {
            "x": 20,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 27, "y": 3, "target_map_name": "SILPH_CO_7F", "target_map_id": 212, "warp_id": 6},
        {"x": 9, "y": 15, "target_map_name": "SILPH_CO_9F", "target_map_id": 233, "warp_id": 5},
        {"x": 11, "y": 5, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 5},
        {"x": 3, "y": 15, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 6},
    ],
    "SILPH_CO_6F": [
        {"x": 16, "y": 0, "target_map_name": "SILPH_CO_7F", "target_map_id": 212, "warp_id": 2},
        {"x": 14, "y": 0, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 1},
        {
            "x": 18,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 3, "y": 3, "target_map_name": "SILPH_CO_4F", "target_map_id": 209, "warp_id": 5},
        {"x": 23, "y": 3, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 7},
    ],
    "SILPH_CO_7F": [
        {"x": 16, "y": 0, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 2},
        {"x": 22, "y": 0, "target_map_name": "SILPH_CO_6F", "target_map_id": 211, "warp_id": 1},
        {
            "x": 18,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 5, "y": 7, "target_map_name": "SILPH_CO_11F", "target_map_id": 235, "warp_id": 4},
        {"x": 5, "y": 3, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 9},
        {"x": 21, "y": 15, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 4},
    ],
    "SILPH_CO_8F": [
        {"x": 16, "y": 0, "target_map_name": "SILPH_CO_9F", "target_map_id": 233, "warp_id": 2},
        {"x": 14, "y": 0, "target_map_name": "SILPH_CO_7F", "target_map_id": 212, "warp_id": 1},
        {
            "x": 18,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 3, "y": 11, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 7},
        {"x": 3, "y": 15, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 5},
        {"x": 11, "y": 5, "target_map_name": "SILPH_CO_2F", "target_map_id": 207, "warp_id": 6},
        {"x": 11, "y": 9, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 4},
    ],
    "SILPH_CO_9F": [
        {"x": 14, "y": 0, "target_map_name": "SILPH_CO_10F", "target_map_id": 234, "warp_id": 1},
        {"x": 16, "y": 0, "target_map_name": "SILPH_CO_8F", "target_map_id": 213, "warp_id": 1},
        {
            "x": 18,
            "y": 0,
            "target_map_name": "SILPH_CO_ELEVATOR",
            "target_map_id": 236,
            "warp_id": 1,
        },
        {"x": 9, "y": 3, "target_map_name": "SILPH_CO_3F", "target_map_id": 208, "warp_id": 8},
        {"x": 17, "y": 15, "target_map_name": "SILPH_CO_5F", "target_map_id": 210, "warp_id": 5},
    ],
    "SILPH_CO_ELEVATOR": [
        {"x": 1, "y": 3, "target_map_name": "UNUSED_MAP_ED", "target_map_id": 237, "warp_id": 1},
        {"x": 2, "y": 3, "target_map_name": "UNUSED_MAP_ED", "target_map_id": 237, "warp_id": 1},
    ],
    "SS_ANNE_1F": [
        {"x": 26, "y": 0, "target_map_name": "VERMILION_DOCK", "target_map_id": 94, "warp_id": 2},
        {"x": 27, "y": 0, "target_map_name": "VERMILION_DOCK", "target_map_id": 94, "warp_id": 2},
        {
            "x": 31,
            "y": 8,
            "target_map_name": "SS_ANNE_1F_ROOMS",
            "target_map_id": 102,
            "warp_id": 1,
        },
        {
            "x": 23,
            "y": 8,
            "target_map_name": "SS_ANNE_1F_ROOMS",
            "target_map_id": 102,
            "warp_id": 2,
        },
        {
            "x": 19,
            "y": 8,
            "target_map_name": "SS_ANNE_1F_ROOMS",
            "target_map_id": 102,
            "warp_id": 3,
        },
        {
            "x": 15,
            "y": 8,
            "target_map_name": "SS_ANNE_1F_ROOMS",
            "target_map_id": 102,
            "warp_id": 4,
        },
        {
            "x": 11,
            "y": 8,
            "target_map_name": "SS_ANNE_1F_ROOMS",
            "target_map_id": 102,
            "warp_id": 5,
        },
        {"x": 7, "y": 8, "target_map_name": "SS_ANNE_1F_ROOMS", "target_map_id": 102, "warp_id": 6},
        {"x": 2, "y": 6, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 7},
        {"x": 37, "y": 15, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 6},
        {"x": 3, "y": 16, "target_map_name": "SS_ANNE_KITCHEN", "target_map_id": 100, "warp_id": 1},
    ],
    "SS_ANNE_1F_ROOMS": [
        {"x": 0, "y": 0, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 3},
        {"x": 10, "y": 0, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 4},
        {"x": 20, "y": 0, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 5},
        {"x": 0, "y": 10, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 6},
        {"x": 10, "y": 10, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 7},
        {"x": 20, "y": 10, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 8},
    ],
    "SS_ANNE_2F": [
        {
            "x": 9,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 1,
        },
        {
            "x": 13,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 3,
        },
        {
            "x": 17,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 5,
        },
        {
            "x": 21,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 7,
        },
        {
            "x": 25,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 9,
        },
        {
            "x": 29,
            "y": 11,
            "target_map_name": "SS_ANNE_2F_ROOMS",
            "target_map_id": 103,
            "warp_id": 11,
        },
        {"x": 2, "y": 4, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 9},
        {"x": 2, "y": 12, "target_map_name": "SS_ANNE_3F", "target_map_id": 97, "warp_id": 2},
        {
            "x": 36,
            "y": 4,
            "target_map_name": "SS_ANNE_CAPTAINS_ROOM",
            "target_map_id": 101,
            "warp_id": 1,
        },
    ],
    "SS_ANNE_2F_ROOMS": [
        {"x": 2, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 1},
        {"x": 3, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 1},
        {"x": 12, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 2},
        {"x": 13, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 2},
        {"x": 22, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 3},
        {"x": 23, "y": 5, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 3},
        {"x": 2, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 4},
        {"x": 3, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 4},
        {"x": 12, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 5},
        {"x": 13, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 5},
        {"x": 22, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 6},
        {"x": 23, "y": 15, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 6},
    ],
    "SS_ANNE_3F": [
        {"x": 0, "y": 3, "target_map_name": "SS_ANNE_BOW", "target_map_id": 99, "warp_id": 1},
        {"x": 19, "y": 3, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 8},
    ],
    "SS_ANNE_B1F": [
        {
            "x": 23,
            "y": 3,
            "target_map_name": "SS_ANNE_B1F_ROOMS",
            "target_map_id": 104,
            "warp_id": 9,
        },
        {
            "x": 19,
            "y": 3,
            "target_map_name": "SS_ANNE_B1F_ROOMS",
            "target_map_id": 104,
            "warp_id": 7,
        },
        {
            "x": 15,
            "y": 3,
            "target_map_name": "SS_ANNE_B1F_ROOMS",
            "target_map_id": 104,
            "warp_id": 5,
        },
        {
            "x": 11,
            "y": 3,
            "target_map_name": "SS_ANNE_B1F_ROOMS",
            "target_map_id": 104,
            "warp_id": 3,
        },
        {
            "x": 7,
            "y": 3,
            "target_map_name": "SS_ANNE_B1F_ROOMS",
            "target_map_id": 104,
            "warp_id": 1,
        },
        {"x": 27, "y": 5, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 10},
    ],
    "SS_ANNE_B1F_ROOMS": [
        {"x": 2, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 5},
        {"x": 3, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 5},
        {"x": 12, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 4},
        {"x": 13, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 4},
        {"x": 22, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 3},
        {"x": 23, "y": 5, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 3},
        {"x": 2, "y": 15, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 2},
        {"x": 3, "y": 15, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 2},
        {"x": 12, "y": 15, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 1},
        {"x": 13, "y": 15, "target_map_name": "SS_ANNE_B1F", "target_map_id": 98, "warp_id": 1},
    ],
    "SS_ANNE_BOW": [
        {"x": 13, "y": 6, "target_map_name": "SS_ANNE_3F", "target_map_id": 97, "warp_id": 1},
        {"x": 13, "y": 7, "target_map_name": "SS_ANNE_3F", "target_map_id": 97, "warp_id": 1},
    ],
    "SS_ANNE_CAPTAINS_ROOM": [
        {"x": 0, "y": 7, "target_map_name": "SS_ANNE_2F", "target_map_id": 96, "warp_id": 9}
    ],
    "SS_ANNE_KITCHEN": [
        {"x": 6, "y": 0, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 11}
    ],
    "TRADE_CENTER": [],
    "UNDERGROUND_PATH_NORTH_SOUTH": [
        {
            "x": 5,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_5",
            "target_map_id": 71,
            "warp_id": 3,
        },
        {
            "x": 2,
            "y": 41,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_6",
            "target_map_id": 74,
            "warp_id": 3,
        },
    ],
    "UNDERGROUND_PATH_ROUTE_5": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {
            "x": 4,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_NORTH_SOUTH",
            "target_map_id": 119,
            "warp_id": 1,
        },
    ],
    "UNDERGROUND_PATH_ROUTE_6": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {
            "x": 4,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_NORTH_SOUTH",
            "target_map_id": 119,
            "warp_id": 2,
        },
    ],
    "UNDERGROUND_PATH_ROUTE_7": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {
            "x": 4,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_WEST_EAST",
            "target_map_id": 121,
            "warp_id": 1,
        },
    ],
    "UNDERGROUND_PATH_ROUTE_7_COPY": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {
            "x": 4,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_WEST_EAST",
            "target_map_id": 121,
            "warp_id": 1,
        },
    ],
    "UNDERGROUND_PATH_ROUTE_8": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {
            "x": 4,
            "y": 4,
            "target_map_name": "UNDERGROUND_PATH_WEST_EAST",
            "target_map_id": 121,
            "warp_id": 2,
        },
    ],
    "UNDERGROUND_PATH_WEST_EAST": [
        {
            "x": 2,
            "y": 5,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_7",
            "target_map_id": 77,
            "warp_id": 3,
        },
        {
            "x": 47,
            "y": 2,
            "target_map_name": "UNDERGROUND_PATH_ROUTE_8",
            "target_map_id": 80,
            "warp_id": 3,
        },
    ],
    "VERMILION_CITY": [
        {
            "x": 11,
            "y": 3,
            "target_map_name": "VERMILION_POKECENTER",
            "target_map_id": 89,
            "warp_id": 1,
        },
        {"x": 9, "y": 13, "target_map_name": "POKEMON_FAN_CLUB", "target_map_id": 90, "warp_id": 1},
        {"x": 23, "y": 13, "target_map_name": "VERMILION_MART", "target_map_id": 91, "warp_id": 1},
        {"x": 12, "y": 19, "target_map_name": "VERMILION_GYM", "target_map_id": 92, "warp_id": 1},
        {
            "x": 23,
            "y": 19,
            "target_map_name": "VERMILION_PIDGEY_HOUSE",
            "target_map_id": 93,
            "warp_id": 1,
        },
        {"x": 18, "y": 31, "target_map_name": "VERMILION_DOCK", "target_map_id": 94, "warp_id": 1},
        {"x": 19, "y": 31, "target_map_name": "VERMILION_DOCK", "target_map_id": 94, "warp_id": 1},
        {
            "x": 15,
            "y": 13,
            "target_map_name": "VERMILION_TRADE_HOUSE",
            "target_map_id": 196,
            "warp_id": 1,
        },
        {
            "x": 7,
            "y": 3,
            "target_map_name": "VERMILION_OLD_ROD_HOUSE",
            "target_map_id": 163,
            "warp_id": 1,
        },
    ],
    "VERMILION_DOCK": [
        {"x": 14, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 14, "y": 2, "target_map_name": "SS_ANNE_1F", "target_map_id": 95, "warp_id": 2},
    ],
    "VERMILION_GYM": [
        {"x": 4, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "VERMILION_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "VERMILION_OLD_ROD_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 9},
    ],
    "VERMILION_PIDGEY_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "VERMILION_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "VERMILION_TRADE_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 8},
    ],
    "VICTORY_ROAD_1F": [
        {"x": 8, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 9, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 1, "y": 1, "target_map_name": "VICTORY_ROAD_2F", "target_map_id": 194, "warp_id": 1},
    ],
    "VICTORY_ROAD_2F": [
        {"x": 0, "y": 8, "target_map_name": "VICTORY_ROAD_1F", "target_map_id": 108, "warp_id": 3},
        {"x": 29, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 29, "y": 8, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 23, "y": 7, "target_map_name": "VICTORY_ROAD_3F", "target_map_id": 198, "warp_id": 1},
        {
            "x": 25,
            "y": 14,
            "target_map_name": "VICTORY_ROAD_3F",
            "target_map_id": 198,
            "warp_id": 3,
        },
        {"x": 27, "y": 7, "target_map_name": "VICTORY_ROAD_3F", "target_map_id": 198, "warp_id": 2},
        {"x": 1, "y": 1, "target_map_name": "VICTORY_ROAD_3F", "target_map_id": 198, "warp_id": 4},
    ],
    "VICTORY_ROAD_3F": [
        {"x": 23, "y": 7, "target_map_name": "VICTORY_ROAD_2F", "target_map_id": 194, "warp_id": 4},
        {"x": 26, "y": 8, "target_map_name": "VICTORY_ROAD_2F", "target_map_id": 194, "warp_id": 6},
        {
            "x": 27,
            "y": 15,
            "target_map_name": "VICTORY_ROAD_2F",
            "target_map_id": 194,
            "warp_id": 5,
        },
        {"x": 2, "y": 0, "target_map_name": "VICTORY_ROAD_2F", "target_map_id": 194, "warp_id": 7},
    ],
    "VIRIDIAN_CITY": [
        {
            "x": 23,
            "y": 25,
            "target_map_name": "VIRIDIAN_POKECENTER",
            "target_map_id": 41,
            "warp_id": 1,
        },
        {"x": 29, "y": 19, "target_map_name": "VIRIDIAN_MART", "target_map_id": 42, "warp_id": 1},
        {
            "x": 21,
            "y": 15,
            "target_map_name": "VIRIDIAN_SCHOOL_HOUSE",
            "target_map_id": 43,
            "warp_id": 1,
        },
        {
            "x": 21,
            "y": 9,
            "target_map_name": "VIRIDIAN_NICKNAME_HOUSE",
            "target_map_id": 44,
            "warp_id": 1,
        },
        {"x": 32, "y": 7, "target_map_name": "VIRIDIAN_GYM", "target_map_id": 45, "warp_id": 1},
    ],
    "VIRIDIAN_FOREST": [
        {
            "x": 1,
            "y": 0,
            "target_map_name": "VIRIDIAN_FOREST_NORTH_GATE",
            "target_map_id": 47,
            "warp_id": 3,
        },
        {
            "x": 2,
            "y": 0,
            "target_map_name": "VIRIDIAN_FOREST_NORTH_GATE",
            "target_map_id": 47,
            "warp_id": 4,
        },
        {
            "x": 15,
            "y": 47,
            "target_map_name": "VIRIDIAN_FOREST_SOUTH_GATE",
            "target_map_id": 50,
            "warp_id": 2,
        },
        {
            "x": 16,
            "y": 47,
            "target_map_name": "VIRIDIAN_FOREST_SOUTH_GATE",
            "target_map_id": 50,
            "warp_id": 2,
        },
        {
            "x": 17,
            "y": 47,
            "target_map_name": "VIRIDIAN_FOREST_SOUTH_GATE",
            "target_map_id": 50,
            "warp_id": 2,
        },
        {
            "x": 18,
            "y": 47,
            "target_map_name": "VIRIDIAN_FOREST_SOUTH_GATE",
            "target_map_id": 50,
            "warp_id": 2,
        },
    ],
    "VIRIDIAN_FOREST_NORTH_GATE": [
        {"x": 4, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 5, "y": 0, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 4, "y": 7, "target_map_name": "VIRIDIAN_FOREST", "target_map_id": 51, "warp_id": 1},
        {"x": 5, "y": 7, "target_map_name": "VIRIDIAN_FOREST", "target_map_id": 51, "warp_id": 1},
    ],
    "VIRIDIAN_FOREST_SOUTH_GATE": [
        {"x": 4, "y": 0, "target_map_name": "VIRIDIAN_FOREST", "target_map_id": 51, "warp_id": 4},
        {"x": 5, "y": 0, "target_map_name": "VIRIDIAN_FOREST", "target_map_id": 51, "warp_id": 5},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 6},
    ],
    "VIRIDIAN_GYM": [
        {"x": 16, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
        {"x": 17, "y": 17, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 5},
    ],
    "VIRIDIAN_MART": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 2},
    ],
    "VIRIDIAN_NICKNAME_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
    "VIRIDIAN_POKECENTER": [
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 1},
    ],
    "VIRIDIAN_SCHOOL_HOUSE": [
        {"x": 2, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
        {"x": 3, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 3},
    ],
    "WARDENS_HOUSE": [
        {"x": 4, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
        {"x": 5, "y": 7, "target_map_name": "LAST_MAP", "target_map_id": 255, "warp_id": 4},
    ],
}

WARP_ID_DICT = {
    "AGATHAS_ROOM@0": 0,
    "AGATHAS_ROOM@1": 1,
    "AGATHAS_ROOM@2": 2,
    "AGATHAS_ROOM@3": 3,
    "BIKE_SHOP@0": 4,
    "BIKE_SHOP@1": 5,
    "BILLS_HOUSE@0": 6,
    "BILLS_HOUSE@1": 7,
    "BLUES_HOUSE@0": 8,
    "BLUES_HOUSE@1": 9,
    "BRUNOS_ROOM@0": 10,
    "BRUNOS_ROOM@1": 11,
    "BRUNOS_ROOM@2": 12,
    "BRUNOS_ROOM@3": 13,
    "CELADON_CHIEF_HOUSE@0": 14,
    "CELADON_CHIEF_HOUSE@1": 15,
    "CELADON_CITY@0": 16,
    "CELADON_CITY@1": 17,
    "CELADON_CITY@2": 18,
    "CELADON_CITY@3": 19,
    "CELADON_CITY@4": 20,
    "CELADON_CITY@5": 21,
    "CELADON_CITY@6": 22,
    "CELADON_CITY@7": 23,
    "CELADON_CITY@8": 24,
    "CELADON_CITY@9": 25,
    "CELADON_CITY@10": 26,
    "CELADON_CITY@11": 27,
    "CELADON_CITY@12": 28,
    "CELADON_DINER@0": 29,
    "CELADON_DINER@1": 30,
    "CELADON_GYM@0": 31,
    "CELADON_GYM@1": 32,
    "CELADON_HOTEL@0": 33,
    "CELADON_HOTEL@1": 34,
    "CELADON_MANSION_1F@0": 35,
    "CELADON_MANSION_1F@1": 36,
    "CELADON_MANSION_1F@2": 37,
    "CELADON_MANSION_1F@3": 38,
    "CELADON_MANSION_1F@4": 39,
    "CELADON_MANSION_2F@0": 40,
    "CELADON_MANSION_2F@1": 41,
    "CELADON_MANSION_2F@2": 42,
    "CELADON_MANSION_2F@3": 43,
    "CELADON_MANSION_3F@0": 44,
    "CELADON_MANSION_3F@1": 45,
    "CELADON_MANSION_3F@2": 46,
    "CELADON_MANSION_3F@3": 47,
    "CELADON_MANSION_ROOF@0": 48,
    "CELADON_MANSION_ROOF@1": 49,
    "CELADON_MANSION_ROOF@2": 50,
    "CELADON_MANSION_ROOF_HOUSE@0": 51,
    "CELADON_MANSION_ROOF_HOUSE@1": 52,
    "CELADON_MART_1F@0": 53,
    "CELADON_MART_1F@1": 54,
    "CELADON_MART_1F@2": 55,
    "CELADON_MART_1F@3": 56,
    "CELADON_MART_1F@4": 57,
    "CELADON_MART_1F@5": 58,
    "CELADON_MART_2F@0": 59,
    "CELADON_MART_2F@1": 60,
    "CELADON_MART_2F@2": 61,
    "CELADON_MART_3F@0": 62,
    "CELADON_MART_3F@1": 63,
    "CELADON_MART_3F@2": 64,
    "CELADON_MART_4F@0": 65,
    "CELADON_MART_4F@1": 66,
    "CELADON_MART_4F@2": 67,
    "CELADON_MART_5F@0": 68,
    "CELADON_MART_5F@1": 69,
    "CELADON_MART_5F@2": 70,
    "CELADON_MART_ELEVATOR@0": 71,
    "CELADON_MART_ELEVATOR@1": 72,
    "CELADON_MART_ROOF@0": 73,
    "CELADON_POKECENTER@0": 74,
    "CELADON_POKECENTER@1": 75,
    "CERULEAN_BADGE_HOUSE@0": 76,
    "CERULEAN_BADGE_HOUSE@1": 77,
    "CERULEAN_BADGE_HOUSE@2": 78,
    "CERULEAN_CAVE_1F@0": 79,
    "CERULEAN_CAVE_1F@1": 80,
    "CERULEAN_CAVE_1F@2": 81,
    "CERULEAN_CAVE_1F@3": 82,
    "CERULEAN_CAVE_1F@4": 83,
    "CERULEAN_CAVE_1F@5": 84,
    "CERULEAN_CAVE_1F@6": 85,
    "CERULEAN_CAVE_1F@7": 86,
    "CERULEAN_CAVE_1F@8": 87,
    "CERULEAN_CAVE_2F@0": 88,
    "CERULEAN_CAVE_2F@1": 89,
    "CERULEAN_CAVE_2F@2": 90,
    "CERULEAN_CAVE_2F@3": 91,
    "CERULEAN_CAVE_2F@4": 92,
    "CERULEAN_CAVE_2F@5": 93,
    "CERULEAN_CAVE_B1F@0": 94,
    "CERULEAN_CITY@0": 95,
    "CERULEAN_CITY@1": 96,
    "CERULEAN_CITY@2": 97,
    "CERULEAN_CITY@3": 98,
    "CERULEAN_CITY@4": 99,
    "CERULEAN_CITY@5": 100,
    "CERULEAN_CITY@6": 101,
    "CERULEAN_CITY@7": 102,
    "CERULEAN_CITY@8": 103,
    "CERULEAN_CITY@9": 104,
    "CERULEAN_GYM@0": 105,
    "CERULEAN_GYM@1": 106,
    "CERULEAN_MART@0": 107,
    "CERULEAN_MART@1": 108,
    "CERULEAN_POKECENTER@0": 109,
    "CERULEAN_POKECENTER@1": 110,
    "CERULEAN_TRADE_HOUSE@0": 111,
    "CERULEAN_TRADE_HOUSE@1": 112,
    "CERULEAN_TRASHED_HOUSE@0": 113,
    "CERULEAN_TRASHED_HOUSE@1": 114,
    "CERULEAN_TRASHED_HOUSE@2": 115,
    "CHAMPIONS_ROOM@0": 116,
    "CHAMPIONS_ROOM@1": 117,
    "CHAMPIONS_ROOM@2": 118,
    "CHAMPIONS_ROOM@3": 119,
    "CINNABAR_GYM@0": 120,
    "CINNABAR_GYM@1": 121,
    "CINNABAR_ISLAND@0": 122,
    "CINNABAR_ISLAND@1": 123,
    "CINNABAR_ISLAND@2": 124,
    "CINNABAR_ISLAND@3": 125,
    "CINNABAR_ISLAND@4": 126,
    "CINNABAR_LAB@0": 127,
    "CINNABAR_LAB@1": 128,
    "CINNABAR_LAB@2": 129,
    "CINNABAR_LAB@3": 130,
    "CINNABAR_LAB@4": 131,
    "CINNABAR_LAB_FOSSIL_ROOM@0": 132,
    "CINNABAR_LAB_FOSSIL_ROOM@1": 133,
    "CINNABAR_LAB_METRONOME_ROOM@0": 134,
    "CINNABAR_LAB_METRONOME_ROOM@1": 135,
    "CINNABAR_LAB_TRADE_ROOM@0": 136,
    "CINNABAR_LAB_TRADE_ROOM@1": 137,
    "CINNABAR_MART@0": 138,
    "CINNABAR_MART@1": 139,
    "CINNABAR_POKECENTER@0": 140,
    "CINNABAR_POKECENTER@1": 141,
    "COLOSSEUM@0": 142,
    "COPYCATS_HOUSE_1F@0": 143,
    "COPYCATS_HOUSE_1F@1": 144,
    "COPYCATS_HOUSE_1F@2": 145,
    "COPYCATS_HOUSE_2F@0": 146,
    "DAYCARE@0": 147,
    "DAYCARE@1": 148,
    "DIGLETTS_CAVE@0": 149,
    "DIGLETTS_CAVE@1": 150,
    "DIGLETTS_CAVE_ROUTE_11@0": 151,
    "DIGLETTS_CAVE_ROUTE_11@1": 152,
    "DIGLETTS_CAVE_ROUTE_11@2": 153,
    "DIGLETTS_CAVE_ROUTE_2@0": 154,
    "DIGLETTS_CAVE_ROUTE_2@1": 155,
    "DIGLETTS_CAVE_ROUTE_2@2": 156,
    "FIGHTING_DOJO@0": 157,
    "FIGHTING_DOJO@1": 158,
    "FUCHSIA_BILLS_GRANDPAS_HOUSE@0": 159,
    "FUCHSIA_BILLS_GRANDPAS_HOUSE@1": 160,
    "FUCHSIA_CITY@0": 161,
    "FUCHSIA_CITY@1": 162,
    "FUCHSIA_CITY@2": 163,
    "FUCHSIA_CITY@3": 164,
    "FUCHSIA_CITY@4": 165,
    "FUCHSIA_CITY@5": 166,
    "FUCHSIA_CITY@6": 167,
    "FUCHSIA_CITY@7": 168,
    "FUCHSIA_CITY@8": 169,
    "FUCHSIA_GOOD_ROD_HOUSE@0": 170,
    "FUCHSIA_GOOD_ROD_HOUSE@1": 171,
    "FUCHSIA_GOOD_ROD_HOUSE@2": 172,
    "FUCHSIA_GYM@0": 173,
    "FUCHSIA_GYM@1": 174,
    "FUCHSIA_MART@0": 175,
    "FUCHSIA_MART@1": 176,
    "FUCHSIA_MEETING_ROOM@0": 177,
    "FUCHSIA_MEETING_ROOM@1": 178,
    "FUCHSIA_POKECENTER@0": 179,
    "FUCHSIA_POKECENTER@1": 180,
    "GAME_CORNER@0": 181,
    "GAME_CORNER@1": 182,
    "GAME_CORNER@2": 183,
    "GAME_CORNER_PRIZE_ROOM@0": 184,
    "GAME_CORNER_PRIZE_ROOM@1": 185,
    "HALL_OF_FAME@0": 186,
    "HALL_OF_FAME@1": 187,
    "INDIGO_PLATEAU@0": 188,
    "INDIGO_PLATEAU@1": 189,
    "INDIGO_PLATEAU_LOBBY@0": 190,
    "INDIGO_PLATEAU_LOBBY@1": 191,
    "INDIGO_PLATEAU_LOBBY@2": 192,
    "LANCES_ROOM@0": 193,
    "LANCES_ROOM@1": 194,
    "LANCES_ROOM@2": 195,
    "LAVENDER_CUBONE_HOUSE@0": 196,
    "LAVENDER_CUBONE_HOUSE@1": 197,
    "LAVENDER_MART@0": 198,
    "LAVENDER_MART@1": 199,
    "LAVENDER_POKECENTER@0": 200,
    "LAVENDER_POKECENTER@1": 201,
    "LAVENDER_TOWN@0": 202,
    "LAVENDER_TOWN@1": 203,
    "LAVENDER_TOWN@2": 204,
    "LAVENDER_TOWN@3": 205,
    "LAVENDER_TOWN@4": 206,
    "LAVENDER_TOWN@5": 207,
    "LORELEIS_ROOM@0": 208,
    "LORELEIS_ROOM@1": 209,
    "LORELEIS_ROOM@2": 210,
    "LORELEIS_ROOM@3": 211,
    "MR_FUJIS_HOUSE@0": 212,
    "MR_FUJIS_HOUSE@1": 213,
    "MR_PSYCHICS_HOUSE@0": 214,
    "MR_PSYCHICS_HOUSE@1": 215,
    "MT_MOON_1F@0": 216,
    "MT_MOON_1F@1": 217,
    "MT_MOON_1F@2": 218,
    "MT_MOON_1F@3": 219,
    "MT_MOON_1F@4": 220,
    "MT_MOON_B1F@0": 221,
    "MT_MOON_B1F@1": 222,
    "MT_MOON_B1F@2": 223,
    "MT_MOON_B1F@3": 224,
    "MT_MOON_B1F@4": 225,
    "MT_MOON_B1F@5": 226,
    "MT_MOON_B1F@6": 227,
    "MT_MOON_B1F@7": 228,
    "MT_MOON_B2F@0": 229,
    "MT_MOON_B2F@1": 230,
    "MT_MOON_B2F@2": 231,
    "MT_MOON_B2F@3": 232,
    "MT_MOON_POKECENTER@0": 233,
    "MT_MOON_POKECENTER@1": 234,
    "MUSEUM_1F@0": 235,
    "MUSEUM_1F@1": 236,
    "MUSEUM_1F@2": 237,
    "MUSEUM_1F@3": 238,
    "MUSEUM_1F@4": 239,
    "MUSEUM_2F@0": 240,
    "NAME_RATERS_HOUSE@0": 241,
    "NAME_RATERS_HOUSE@1": 242,
    "OAKS_LAB@0": 243,
    "OAKS_LAB@1": 244,
    "PALLET_TOWN@0": 245,
    "PALLET_TOWN@1": 246,
    "PALLET_TOWN@2": 247,
    "PEWTER_CITY@0": 248,
    "PEWTER_CITY@1": 249,
    "PEWTER_CITY@2": 250,
    "PEWTER_CITY@3": 251,
    "PEWTER_CITY@4": 252,
    "PEWTER_CITY@5": 253,
    "PEWTER_CITY@6": 254,
    "PEWTER_GYM@0": 255,
    "PEWTER_GYM@1": 256,
    "PEWTER_MART@0": 257,
    "PEWTER_MART@1": 258,
    "PEWTER_NIDORAN_HOUSE@0": 259,
    "PEWTER_NIDORAN_HOUSE@1": 260,
    "PEWTER_POKECENTER@0": 261,
    "PEWTER_POKECENTER@1": 262,
    "PEWTER_SPEECH_HOUSE@0": 263,
    "PEWTER_SPEECH_HOUSE@1": 264,
    "POKEMON_FAN_CLUB@0": 265,
    "POKEMON_FAN_CLUB@1": 266,
    "POKEMON_MANSION_1F@0": 267,
    "POKEMON_MANSION_1F@1": 268,
    "POKEMON_MANSION_1F@2": 269,
    "POKEMON_MANSION_1F@3": 270,
    "POKEMON_MANSION_1F@4": 271,
    "POKEMON_MANSION_1F@5": 272,
    "POKEMON_MANSION_1F@6": 273,
    "POKEMON_MANSION_1F@7": 274,
    "POKEMON_MANSION_2F@0": 275,
    "POKEMON_MANSION_2F@1": 276,
    "POKEMON_MANSION_2F@2": 277,
    "POKEMON_MANSION_2F@3": 278,
    "POKEMON_MANSION_3F@0": 279,
    "POKEMON_MANSION_3F@1": 280,
    "POKEMON_MANSION_3F@2": 281,
    "POKEMON_MANSION_B1F@0": 282,
    "POKEMON_TOWER_1F@0": 283,
    "POKEMON_TOWER_1F@1": 284,
    "POKEMON_TOWER_1F@2": 285,
    "POKEMON_TOWER_2F@0": 286,
    "POKEMON_TOWER_2F@1": 287,
    "POKEMON_TOWER_3F@0": 288,
    "POKEMON_TOWER_3F@1": 289,
    "POKEMON_TOWER_4F@0": 290,
    "POKEMON_TOWER_4F@1": 291,
    "POKEMON_TOWER_5F@0": 292,
    "POKEMON_TOWER_5F@1": 293,
    "POKEMON_TOWER_6F@0": 294,
    "POKEMON_TOWER_6F@1": 295,
    "POKEMON_TOWER_7F@0": 296,
    "POWER_PLANT@0": 297,
    "POWER_PLANT@1": 298,
    "POWER_PLANT@2": 299,
    "REDS_HOUSE_1F@0": 300,
    "REDS_HOUSE_1F@1": 301,
    "REDS_HOUSE_1F@2": 302,
    "REDS_HOUSE_2F@0": 303,
    "ROCKET_HIDEOUT_B1F@0": 304,
    "ROCKET_HIDEOUT_B1F@1": 305,
    "ROCKET_HIDEOUT_B1F@2": 306,
    "ROCKET_HIDEOUT_B1F@3": 307,
    "ROCKET_HIDEOUT_B1F@4": 308,
    "ROCKET_HIDEOUT_B2F@0": 309,
    "ROCKET_HIDEOUT_B2F@1": 310,
    "ROCKET_HIDEOUT_B2F@2": 311,
    "ROCKET_HIDEOUT_B2F@3": 312,
    "ROCKET_HIDEOUT_B2F@4": 313,
    "ROCKET_HIDEOUT_B3F@0": 314,
    "ROCKET_HIDEOUT_B3F@1": 315,
    "ROCKET_HIDEOUT_B4F@0": 316,
    "ROCKET_HIDEOUT_B4F@1": 317,
    "ROCKET_HIDEOUT_B4F@2": 318,
    "ROCKET_HIDEOUT_ELEVATOR@0": 319,
    "ROCKET_HIDEOUT_ELEVATOR@1": 320,
    "ROCK_TUNNEL_1F@0": 321,
    "ROCK_TUNNEL_1F@1": 322,
    "ROCK_TUNNEL_1F@2": 323,
    "ROCK_TUNNEL_1F@3": 324,
    "ROCK_TUNNEL_1F@4": 325,
    "ROCK_TUNNEL_1F@5": 326,
    "ROCK_TUNNEL_1F@6": 327,
    "ROCK_TUNNEL_1F@7": 328,
    "ROCK_TUNNEL_B1F@0": 329,
    "ROCK_TUNNEL_B1F@1": 330,
    "ROCK_TUNNEL_B1F@2": 331,
    "ROCK_TUNNEL_B1F@3": 332,
    "ROCK_TUNNEL_POKECENTER@0": 333,
    "ROCK_TUNNEL_POKECENTER@1": 334,
    "ROUTE_1@0": 335,
    "ROUTE_10@0": 336,
    "ROUTE_10@1": 337,
    "ROUTE_10@2": 338,
    "ROUTE_10@3": 339,
    "ROUTE_11@0": 340,
    "ROUTE_11@1": 341,
    "ROUTE_11@2": 342,
    "ROUTE_11@3": 343,
    "ROUTE_11@4": 344,
    "ROUTE_11_GATE_1F@0": 345,
    "ROUTE_11_GATE_1F@1": 346,
    "ROUTE_11_GATE_1F@2": 347,
    "ROUTE_11_GATE_1F@3": 348,
    "ROUTE_11_GATE_1F@4": 349,
    "ROUTE_11_GATE_2F@0": 350,
    "ROUTE_12@0": 351,
    "ROUTE_12@1": 352,
    "ROUTE_12@2": 353,
    "ROUTE_12@3": 354,
    "ROUTE_12_GATE_1F@0": 355,
    "ROUTE_12_GATE_1F@1": 356,
    "ROUTE_12_GATE_1F@2": 357,
    "ROUTE_12_GATE_1F@3": 358,
    "ROUTE_12_GATE_1F@4": 359,
    "ROUTE_12_GATE_2F@0": 360,
    "ROUTE_12_SUPER_ROD_HOUSE@0": 361,
    "ROUTE_12_SUPER_ROD_HOUSE@1": 362,
    "ROUTE_13@0": 363,
    "ROUTE_14@0": 364,
    "ROUTE_15@0": 365,
    "ROUTE_15@1": 366,
    "ROUTE_15@2": 367,
    "ROUTE_15@3": 368,
    "ROUTE_15_GATE_1F@0": 369,
    "ROUTE_15_GATE_1F@1": 370,
    "ROUTE_15_GATE_1F@2": 371,
    "ROUTE_15_GATE_1F@3": 372,
    "ROUTE_15_GATE_1F@4": 373,
    "ROUTE_15_GATE_2F@0": 374,
    "ROUTE_16@0": 375,
    "ROUTE_16@1": 376,
    "ROUTE_16@2": 377,
    "ROUTE_16@3": 378,
    "ROUTE_16@4": 379,
    "ROUTE_16@5": 380,
    "ROUTE_16@6": 381,
    "ROUTE_16@7": 382,
    "ROUTE_16@8": 383,
    "ROUTE_16_FLY_HOUSE@0": 384,
    "ROUTE_16_FLY_HOUSE@1": 385,
    "ROUTE_16_GATE_1F@0": 386,
    "ROUTE_16_GATE_1F@1": 387,
    "ROUTE_16_GATE_1F@2": 388,
    "ROUTE_16_GATE_1F@3": 389,
    "ROUTE_16_GATE_1F@4": 390,
    "ROUTE_16_GATE_1F@5": 391,
    "ROUTE_16_GATE_1F@6": 392,
    "ROUTE_16_GATE_1F@7": 393,
    "ROUTE_16_GATE_1F@8": 394,
    "ROUTE_16_GATE_2F@0": 395,
    "ROUTE_17@0": 396,
    "ROUTE_18@0": 397,
    "ROUTE_18@1": 398,
    "ROUTE_18@2": 399,
    "ROUTE_18@3": 400,
    "ROUTE_18_GATE_1F@0": 401,
    "ROUTE_18_GATE_1F@1": 402,
    "ROUTE_18_GATE_1F@2": 403,
    "ROUTE_18_GATE_1F@3": 404,
    "ROUTE_18_GATE_1F@4": 405,
    "ROUTE_18_GATE_2F@0": 406,
    "ROUTE_19@0": 407,
    "ROUTE_2@0": 408,
    "ROUTE_2@1": 409,
    "ROUTE_2@2": 410,
    "ROUTE_2@3": 411,
    "ROUTE_2@4": 412,
    "ROUTE_2@5": 413,
    "ROUTE_20@0": 414,
    "ROUTE_20@1": 415,
    "ROUTE_21@0": 416,
    "ROUTE_22@0": 417,
    "ROUTE_22_GATE@0": 418,
    "ROUTE_22_GATE@1": 419,
    "ROUTE_22_GATE@2": 420,
    "ROUTE_22_GATE@3": 421,
    "ROUTE_23@0": 422,
    "ROUTE_23@1": 423,
    "ROUTE_23@2": 424,
    "ROUTE_23@3": 425,
    "ROUTE_24@0": 426,
    "ROUTE_25@0": 427,
    "ROUTE_2_GATE@0": 428,
    "ROUTE_2_GATE@1": 429,
    "ROUTE_2_GATE@2": 430,
    "ROUTE_2_GATE@3": 431,
    "ROUTE_2_TRADE_HOUSE@0": 432,
    "ROUTE_2_TRADE_HOUSE@1": 433,
    "ROUTE_3@0": 434,
    "ROUTE_4@0": 435,
    "ROUTE_4@1": 436,
    "ROUTE_4@2": 437,
    "ROUTE_5@0": 438,
    "ROUTE_5@1": 439,
    "ROUTE_5@2": 440,
    "ROUTE_5@3": 441,
    "ROUTE_5@4": 442,
    "ROUTE_5_GATE@0": 443,
    "ROUTE_5_GATE@1": 444,
    "ROUTE_5_GATE@2": 445,
    "ROUTE_5_GATE@3": 446,
    "ROUTE_6@0": 447,
    "ROUTE_6@1": 448,
    "ROUTE_6@2": 449,
    "ROUTE_6@3": 450,
    "ROUTE_6_GATE@0": 451,
    "ROUTE_6_GATE@1": 452,
    "ROUTE_6_GATE@2": 453,
    "ROUTE_6_GATE@3": 454,
    "ROUTE_7@0": 455,
    "ROUTE_7@1": 456,
    "ROUTE_7@2": 457,
    "ROUTE_7@3": 458,
    "ROUTE_7@4": 459,
    "ROUTE_7_GATE@0": 460,
    "ROUTE_7_GATE@1": 461,
    "ROUTE_7_GATE@2": 462,
    "ROUTE_7_GATE@3": 463,
    "ROUTE_8@0": 464,
    "ROUTE_8@1": 465,
    "ROUTE_8@2": 466,
    "ROUTE_8@3": 467,
    "ROUTE_8@4": 468,
    "ROUTE_8_GATE@0": 469,
    "ROUTE_8_GATE@1": 470,
    "ROUTE_8_GATE@2": 471,
    "ROUTE_8_GATE@3": 472,
    "ROUTE_9@0": 473,
    "SAFARI_ZONE_CENTER@0": 474,
    "SAFARI_ZONE_CENTER@1": 475,
    "SAFARI_ZONE_CENTER@2": 476,
    "SAFARI_ZONE_CENTER@3": 477,
    "SAFARI_ZONE_CENTER@4": 478,
    "SAFARI_ZONE_CENTER@5": 479,
    "SAFARI_ZONE_CENTER@6": 480,
    "SAFARI_ZONE_CENTER@7": 481,
    "SAFARI_ZONE_CENTER@8": 482,
    "SAFARI_ZONE_CENTER_REST_HOUSE@0": 483,
    "SAFARI_ZONE_CENTER_REST_HOUSE@1": 484,
    "SAFARI_ZONE_EAST@0": 485,
    "SAFARI_ZONE_EAST@1": 486,
    "SAFARI_ZONE_EAST@2": 487,
    "SAFARI_ZONE_EAST@3": 488,
    "SAFARI_ZONE_EAST@4": 489,
    "SAFARI_ZONE_EAST_REST_HOUSE@0": 490,
    "SAFARI_ZONE_EAST_REST_HOUSE@1": 491,
    "SAFARI_ZONE_GATE@0": 492,
    "SAFARI_ZONE_GATE@1": 493,
    "SAFARI_ZONE_GATE@2": 494,
    "SAFARI_ZONE_GATE@3": 495,
    "SAFARI_ZONE_NORTH@0": 496,
    "SAFARI_ZONE_NORTH@1": 497,
    "SAFARI_ZONE_NORTH@2": 498,
    "SAFARI_ZONE_NORTH@3": 499,
    "SAFARI_ZONE_NORTH@4": 500,
    "SAFARI_ZONE_NORTH@5": 501,
    "SAFARI_ZONE_NORTH@6": 502,
    "SAFARI_ZONE_NORTH@7": 503,
    "SAFARI_ZONE_NORTH@8": 504,
    "SAFARI_ZONE_NORTH_REST_HOUSE@0": 505,
    "SAFARI_ZONE_NORTH_REST_HOUSE@1": 506,
    "SAFARI_ZONE_SECRET_HOUSE@0": 507,
    "SAFARI_ZONE_SECRET_HOUSE@1": 508,
    "SAFARI_ZONE_WEST@0": 509,
    "SAFARI_ZONE_WEST@1": 510,
    "SAFARI_ZONE_WEST@2": 511,
    "SAFARI_ZONE_WEST@3": 512,
    "SAFARI_ZONE_WEST@4": 513,
    "SAFARI_ZONE_WEST@5": 514,
    "SAFARI_ZONE_WEST@6": 515,
    "SAFARI_ZONE_WEST@7": 516,
    "SAFARI_ZONE_WEST_REST_HOUSE@0": 517,
    "SAFARI_ZONE_WEST_REST_HOUSE@1": 518,
    "SAFFRON_CITY@0": 519,
    "SAFFRON_CITY@1": 520,
    "SAFFRON_CITY@2": 521,
    "SAFFRON_CITY@3": 522,
    "SAFFRON_CITY@4": 523,
    "SAFFRON_CITY@5": 524,
    "SAFFRON_CITY@6": 525,
    "SAFFRON_CITY@7": 526,
    "SAFFRON_GYM@0": 527,
    "SAFFRON_GYM@1": 528,
    "SAFFRON_GYM@2": 529,
    "SAFFRON_GYM@3": 530,
    "SAFFRON_GYM@4": 531,
    "SAFFRON_GYM@5": 532,
    "SAFFRON_GYM@6": 533,
    "SAFFRON_GYM@7": 534,
    "SAFFRON_GYM@8": 535,
    "SAFFRON_GYM@9": 536,
    "SAFFRON_GYM@10": 537,
    "SAFFRON_GYM@11": 538,
    "SAFFRON_GYM@12": 539,
    "SAFFRON_GYM@13": 540,
    "SAFFRON_GYM@14": 541,
    "SAFFRON_GYM@15": 542,
    "SAFFRON_GYM@16": 543,
    "SAFFRON_GYM@17": 544,
    "SAFFRON_GYM@18": 545,
    "SAFFRON_GYM@19": 546,
    "SAFFRON_GYM@20": 547,
    "SAFFRON_GYM@21": 548,
    "SAFFRON_GYM@22": 549,
    "SAFFRON_GYM@23": 550,
    "SAFFRON_GYM@24": 551,
    "SAFFRON_GYM@25": 552,
    "SAFFRON_GYM@26": 553,
    "SAFFRON_GYM@27": 554,
    "SAFFRON_GYM@28": 555,
    "SAFFRON_GYM@29": 556,
    "SAFFRON_GYM@30": 557,
    "SAFFRON_GYM@31": 558,
    "SAFFRON_MART@0": 559,
    "SAFFRON_MART@1": 560,
    "SAFFRON_PIDGEY_HOUSE@0": 561,
    "SAFFRON_PIDGEY_HOUSE@1": 562,
    "SAFFRON_POKECENTER@0": 563,
    "SAFFRON_POKECENTER@1": 564,
    "SEAFOAM_ISLANDS_1F@0": 565,
    "SEAFOAM_ISLANDS_1F@1": 566,
    "SEAFOAM_ISLANDS_1F@2": 567,
    "SEAFOAM_ISLANDS_1F@3": 568,
    "SEAFOAM_ISLANDS_1F@4": 569,
    "SEAFOAM_ISLANDS_1F@5": 570,
    "SEAFOAM_ISLANDS_1F@6": 571,
    "SEAFOAM_ISLANDS_B1F@0": 572,
    "SEAFOAM_ISLANDS_B1F@1": 573,
    "SEAFOAM_ISLANDS_B1F@2": 574,
    "SEAFOAM_ISLANDS_B1F@3": 575,
    "SEAFOAM_ISLANDS_B1F@4": 576,
    "SEAFOAM_ISLANDS_B1F@5": 577,
    "SEAFOAM_ISLANDS_B1F@6": 578,
    "SEAFOAM_ISLANDS_B2F@0": 579,
    "SEAFOAM_ISLANDS_B2F@1": 580,
    "SEAFOAM_ISLANDS_B2F@2": 581,
    "SEAFOAM_ISLANDS_B2F@3": 582,
    "SEAFOAM_ISLANDS_B2F@4": 583,
    "SEAFOAM_ISLANDS_B2F@5": 584,
    "SEAFOAM_ISLANDS_B2F@6": 585,
    "SEAFOAM_ISLANDS_B3F@0": 586,
    "SEAFOAM_ISLANDS_B3F@1": 587,
    "SEAFOAM_ISLANDS_B3F@2": 588,
    "SEAFOAM_ISLANDS_B3F@3": 589,
    "SEAFOAM_ISLANDS_B3F@4": 590,
    "SEAFOAM_ISLANDS_B3F@5": 591,
    "SEAFOAM_ISLANDS_B3F@6": 592,
    "SEAFOAM_ISLANDS_B4F@0": 593,
    "SEAFOAM_ISLANDS_B4F@1": 594,
    "SEAFOAM_ISLANDS_B4F@2": 595,
    "SEAFOAM_ISLANDS_B4F@3": 596,
    "SILPH_CO_10F@0": 597,
    "SILPH_CO_10F@1": 598,
    "SILPH_CO_10F@2": 599,
    "SILPH_CO_10F@3": 600,
    "SILPH_CO_10F@4": 601,
    "SILPH_CO_10F@5": 602,
    "SILPH_CO_11F@0": 603,
    "SILPH_CO_11F@1": 604,
    "SILPH_CO_11F@2": 605,
    "SILPH_CO_11F@3": 606,
    "SILPH_CO_1F@0": 607,
    "SILPH_CO_1F@1": 608,
    "SILPH_CO_1F@2": 609,
    "SILPH_CO_1F@3": 610,
    "SILPH_CO_1F@4": 611,
    "SILPH_CO_2F@0": 612,
    "SILPH_CO_2F@1": 613,
    "SILPH_CO_2F@2": 614,
    "SILPH_CO_2F@3": 615,
    "SILPH_CO_2F@4": 616,
    "SILPH_CO_2F@5": 617,
    "SILPH_CO_2F@6": 618,
    "SILPH_CO_3F@0": 619,
    "SILPH_CO_3F@1": 620,
    "SILPH_CO_3F@2": 621,
    "SILPH_CO_3F@3": 622,
    "SILPH_CO_3F@4": 623,
    "SILPH_CO_3F@5": 624,
    "SILPH_CO_3F@6": 625,
    "SILPH_CO_3F@7": 626,
    "SILPH_CO_3F@8": 627,
    "SILPH_CO_3F@9": 628,
    "SILPH_CO_4F@0": 629,
    "SILPH_CO_4F@1": 630,
    "SILPH_CO_4F@2": 631,
    "SILPH_CO_4F@3": 632,
    "SILPH_CO_4F@4": 633,
    "SILPH_CO_4F@5": 634,
    "SILPH_CO_4F@6": 635,
    "SILPH_CO_5F@0": 636,
    "SILPH_CO_5F@1": 637,
    "SILPH_CO_5F@2": 638,
    "SILPH_CO_5F@3": 639,
    "SILPH_CO_5F@4": 640,
    "SILPH_CO_5F@5": 641,
    "SILPH_CO_5F@6": 642,
    "SILPH_CO_6F@0": 643,
    "SILPH_CO_6F@1": 644,
    "SILPH_CO_6F@2": 645,
    "SILPH_CO_6F@3": 646,
    "SILPH_CO_6F@4": 647,
    "SILPH_CO_7F@0": 648,
    "SILPH_CO_7F@1": 649,
    "SILPH_CO_7F@2": 650,
    "SILPH_CO_7F@3": 651,
    "SILPH_CO_7F@4": 652,
    "SILPH_CO_7F@5": 653,
    "SILPH_CO_8F@0": 654,
    "SILPH_CO_8F@1": 655,
    "SILPH_CO_8F@2": 656,
    "SILPH_CO_8F@3": 657,
    "SILPH_CO_8F@4": 658,
    "SILPH_CO_8F@5": 659,
    "SILPH_CO_8F@6": 660,
    "SILPH_CO_9F@0": 661,
    "SILPH_CO_9F@1": 662,
    "SILPH_CO_9F@2": 663,
    "SILPH_CO_9F@3": 664,
    "SILPH_CO_9F@4": 665,
    "SILPH_CO_ELEVATOR@0": 666,
    "SILPH_CO_ELEVATOR@1": 667,
    "SS_ANNE_1F@0": 668,
    "SS_ANNE_1F@1": 669,
    "SS_ANNE_1F@2": 670,
    "SS_ANNE_1F@3": 671,
    "SS_ANNE_1F@4": 672,
    "SS_ANNE_1F@5": 673,
    "SS_ANNE_1F@6": 674,
    "SS_ANNE_1F@7": 675,
    "SS_ANNE_1F@8": 676,
    "SS_ANNE_1F@9": 677,
    "SS_ANNE_1F@10": 678,
    "SS_ANNE_1F_ROOMS@0": 679,
    "SS_ANNE_1F_ROOMS@1": 680,
    "SS_ANNE_1F_ROOMS@2": 681,
    "SS_ANNE_1F_ROOMS@3": 682,
    "SS_ANNE_1F_ROOMS@4": 683,
    "SS_ANNE_1F_ROOMS@5": 684,
    "SS_ANNE_2F@0": 685,
    "SS_ANNE_2F@1": 686,
    "SS_ANNE_2F@2": 687,
    "SS_ANNE_2F@3": 688,
    "SS_ANNE_2F@4": 689,
    "SS_ANNE_2F@5": 690,
    "SS_ANNE_2F@6": 691,
    "SS_ANNE_2F@7": 692,
    "SS_ANNE_2F@8": 693,
    "SS_ANNE_2F_ROOMS@0": 694,
    "SS_ANNE_2F_ROOMS@1": 695,
    "SS_ANNE_2F_ROOMS@2": 696,
    "SS_ANNE_2F_ROOMS@3": 697,
    "SS_ANNE_2F_ROOMS@4": 698,
    "SS_ANNE_2F_ROOMS@5": 699,
    "SS_ANNE_2F_ROOMS@6": 700,
    "SS_ANNE_2F_ROOMS@7": 701,
    "SS_ANNE_2F_ROOMS@8": 702,
    "SS_ANNE_2F_ROOMS@9": 703,
    "SS_ANNE_2F_ROOMS@10": 704,
    "SS_ANNE_2F_ROOMS@11": 705,
    "SS_ANNE_3F@0": 706,
    "SS_ANNE_3F@1": 707,
    "SS_ANNE_B1F@0": 708,
    "SS_ANNE_B1F@1": 709,
    "SS_ANNE_B1F@2": 710,
    "SS_ANNE_B1F@3": 711,
    "SS_ANNE_B1F@4": 712,
    "SS_ANNE_B1F@5": 713,
    "SS_ANNE_B1F_ROOMS@0": 714,
    "SS_ANNE_B1F_ROOMS@1": 715,
    "SS_ANNE_B1F_ROOMS@2": 716,
    "SS_ANNE_B1F_ROOMS@3": 717,
    "SS_ANNE_B1F_ROOMS@4": 718,
    "SS_ANNE_B1F_ROOMS@5": 719,
    "SS_ANNE_B1F_ROOMS@6": 720,
    "SS_ANNE_B1F_ROOMS@7": 721,
    "SS_ANNE_B1F_ROOMS@8": 722,
    "SS_ANNE_B1F_ROOMS@9": 723,
    "SS_ANNE_BOW@0": 724,
    "SS_ANNE_BOW@1": 725,
    "SS_ANNE_CAPTAINS_ROOM@0": 726,
    "SS_ANNE_KITCHEN@0": 727,
    "TRADE_CENTER@0": 728,
    "UNDERGROUND_PATH_NORTH_SOUTH@0": 729,
    "UNDERGROUND_PATH_NORTH_SOUTH@1": 730,
    "UNDERGROUND_PATH_ROUTE_5@0": 731,
    "UNDERGROUND_PATH_ROUTE_5@1": 732,
    "UNDERGROUND_PATH_ROUTE_5@2": 733,
    "UNDERGROUND_PATH_ROUTE_6@0": 734,
    "UNDERGROUND_PATH_ROUTE_6@1": 735,
    "UNDERGROUND_PATH_ROUTE_6@2": 736,
    "UNDERGROUND_PATH_ROUTE_7@0": 737,
    "UNDERGROUND_PATH_ROUTE_7@1": 738,
    "UNDERGROUND_PATH_ROUTE_7@2": 739,
    "UNDERGROUND_PATH_ROUTE_7_COPY@0": 740,
    "UNDERGROUND_PATH_ROUTE_7_COPY@1": 741,
    "UNDERGROUND_PATH_ROUTE_7_COPY@2": 742,
    "UNDERGROUND_PATH_ROUTE_8@0": 743,
    "UNDERGROUND_PATH_ROUTE_8@1": 744,
    "UNDERGROUND_PATH_ROUTE_8@2": 745,
    "UNDERGROUND_PATH_WEST_EAST@0": 746,
    "UNDERGROUND_PATH_WEST_EAST@1": 747,
    "VERMILION_CITY@0": 748,
    "VERMILION_CITY@1": 749,
    "VERMILION_CITY@2": 750,
    "VERMILION_CITY@3": 751,
    "VERMILION_CITY@4": 752,
    "VERMILION_CITY@5": 753,
    "VERMILION_CITY@6": 754,
    "VERMILION_CITY@7": 755,
    "VERMILION_CITY@8": 756,
    "VERMILION_DOCK@0": 757,
    "VERMILION_DOCK@1": 758,
    "VERMILION_GYM@0": 759,
    "VERMILION_GYM@1": 760,
    "VERMILION_MART@0": 761,
    "VERMILION_MART@1": 762,
    "VERMILION_OLD_ROD_HOUSE@0": 763,
    "VERMILION_OLD_ROD_HOUSE@1": 764,
    "VERMILION_PIDGEY_HOUSE@0": 765,
    "VERMILION_PIDGEY_HOUSE@1": 766,
    "VERMILION_POKECENTER@0": 767,
    "VERMILION_POKECENTER@1": 768,
    "VERMILION_TRADE_HOUSE@0": 769,
    "VERMILION_TRADE_HOUSE@1": 770,
    "VICTORY_ROAD_1F@0": 771,
    "VICTORY_ROAD_1F@1": 772,
    "VICTORY_ROAD_1F@2": 773,
    "VICTORY_ROAD_2F@0": 774,
    "VICTORY_ROAD_2F@1": 775,
    "VICTORY_ROAD_2F@2": 776,
    "VICTORY_ROAD_2F@3": 777,
    "VICTORY_ROAD_2F@4": 778,
    "VICTORY_ROAD_2F@5": 779,
    "VICTORY_ROAD_2F@6": 780,
    "VICTORY_ROAD_3F@0": 781,
    "VICTORY_ROAD_3F@1": 782,
    "VICTORY_ROAD_3F@2": 783,
    "VICTORY_ROAD_3F@3": 784,
    "VIRIDIAN_CITY@0": 785,
    "VIRIDIAN_CITY@1": 786,
    "VIRIDIAN_CITY@2": 787,
    "VIRIDIAN_CITY@3": 788,
    "VIRIDIAN_CITY@4": 789,
    "VIRIDIAN_FOREST@0": 790,
    "VIRIDIAN_FOREST@1": 791,
    "VIRIDIAN_FOREST@2": 792,
    "VIRIDIAN_FOREST@3": 793,
    "VIRIDIAN_FOREST@4": 794,
    "VIRIDIAN_FOREST@5": 795,
    "VIRIDIAN_FOREST_NORTH_GATE@0": 796,
    "VIRIDIAN_FOREST_NORTH_GATE@1": 797,
    "VIRIDIAN_FOREST_NORTH_GATE@2": 798,
    "VIRIDIAN_FOREST_NORTH_GATE@3": 799,
    "VIRIDIAN_FOREST_SOUTH_GATE@0": 800,
    "VIRIDIAN_FOREST_SOUTH_GATE@1": 801,
    "VIRIDIAN_FOREST_SOUTH_GATE@2": 802,
    "VIRIDIAN_FOREST_SOUTH_GATE@3": 803,
    "VIRIDIAN_GYM@0": 804,
    "VIRIDIAN_GYM@1": 805,
    "VIRIDIAN_MART@0": 806,
    "VIRIDIAN_MART@1": 807,
    "VIRIDIAN_NICKNAME_HOUSE@0": 808,
    "VIRIDIAN_NICKNAME_HOUSE@1": 809,
    "VIRIDIAN_POKECENTER@0": 810,
    "VIRIDIAN_POKECENTER@1": 811,
    "VIRIDIAN_SCHOOL_HOUSE@0": 812,
    "VIRIDIAN_SCHOOL_HOUSE@1": 813,
    "WARDENS_HOUSE@0": 814,
    "WARDENS_HOUSE@1": 815,
    "ROUTE_22@1": 816,
}

BASE_STATS = {
    "ABRA": {"hp": 25, "atk": 20, "def": 15, "spd": 90, "spc": 105},
    "AERODACTYL": {"hp": 80, "atk": 105, "def": 65, "spd": 130, "spc": 60},
    "ALAKAZAM": {"hp": 55, "atk": 50, "def": 45, "spd": 120, "spc": 135},
    "ARBOK": {"hp": 60, "atk": 85, "def": 69, "spd": 80, "spc": 65},
    "ARCANINE": {"hp": 90, "atk": 110, "def": 80, "spd": 95, "spc": 80},
    "ARTICUNO": {"hp": 90, "atk": 85, "def": 100, "spd": 85, "spc": 125},
    "BEEDRILL": {"hp": 65, "atk": 80, "def": 40, "spd": 75, "spc": 45},
    "BELLSPROUT": {"hp": 50, "atk": 75, "def": 35, "spd": 40, "spc": 70},
    "BLASTOISE": {"hp": 79, "atk": 83, "def": 100, "spd": 78, "spc": 85},
    "BULBASAUR": {"hp": 45, "atk": 49, "def": 49, "spd": 45, "spc": 65},
    "BUTTERFREE": {"hp": 60, "atk": 45, "def": 50, "spd": 70, "spc": 80},
    "CATERPIE": {"hp": 45, "atk": 30, "def": 35, "spd": 45, "spc": 20},
    "CHANSEY": {"hp": 250, "atk": 5, "def": 5, "spd": 50, "spc": 105},
    "CHARIZARD": {"hp": 78, "atk": 84, "def": 78, "spd": 100, "spc": 85},
    "CHARMANDER": {"hp": 39, "atk": 52, "def": 43, "spd": 65, "spc": 50},
    "CHARMELEON": {"hp": 58, "atk": 64, "def": 58, "spd": 80, "spc": 65},
    "CLEFABLE": {"hp": 95, "atk": 70, "def": 73, "spd": 60, "spc": 85},
    "CLEFAIRY": {"hp": 70, "atk": 45, "def": 48, "spd": 35, "spc": 60},
    "CLOYSTER": {"hp": 50, "atk": 95, "def": 180, "spd": 70, "spc": 85},
    "CUBONE": {"hp": 50, "atk": 50, "def": 95, "spd": 35, "spc": 40},
    "DEWGONG": {"hp": 90, "atk": 70, "def": 80, "spd": 70, "spc": 95},
    "DIGLETT": {"hp": 10, "atk": 55, "def": 25, "spd": 95, "spc": 45},
    "DITTO": {"hp": 48, "atk": 48, "def": 48, "spd": 48, "spc": 48},
    "DODRIO": {"hp": 60, "atk": 110, "def": 70, "spd": 100, "spc": 60},
    "DODUO": {"hp": 35, "atk": 85, "def": 45, "spd": 75, "spc": 35},
    "DRAGONAIR": {"hp": 61, "atk": 84, "def": 65, "spd": 70, "spc": 70},
    "DRAGONITE": {"hp": 91, "atk": 134, "def": 95, "spd": 80, "spc": 100},
    "DRATINI": {"hp": 41, "atk": 64, "def": 45, "spd": 50, "spc": 50},
    "DROWZEE": {"hp": 60, "atk": 48, "def": 45, "spd": 42, "spc": 90},
    "DUGTRIO": {"hp": 35, "atk": 80, "def": 50, "spd": 120, "spc": 70},
    "EEVEE": {"hp": 55, "atk": 55, "def": 50, "spd": 55, "spc": 65},
    "EKANS": {"hp": 35, "atk": 60, "def": 44, "spd": 55, "spc": 40},
    "ELECTABUZZ": {"hp": 65, "atk": 83, "def": 57, "spd": 105, "spc": 85},
    "ELECTRODE": {"hp": 60, "atk": 50, "def": 70, "spd": 140, "spc": 80},
    "EXEGGCUTE": {"hp": 60, "atk": 40, "def": 80, "spd": 40, "spc": 60},
    "EXEGGUTOR": {"hp": 95, "atk": 95, "def": 85, "spd": 55, "spc": 125},
    "FARFETCHD": {"hp": 52, "atk": 65, "def": 55, "spd": 60, "spc": 58},
    "FEAROW": {"hp": 65, "atk": 90, "def": 65, "spd": 100, "spc": 61},
    "FLAREON": {"hp": 65, "atk": 130, "def": 60, "spd": 65, "spc": 110},
    "GASTLY": {"hp": 30, "atk": 35, "def": 30, "spd": 80, "spc": 100},
    "GENGAR": {"hp": 60, "atk": 65, "def": 60, "spd": 110, "spc": 130},
    "GEODUDE": {"hp": 40, "atk": 80, "def": 100, "spd": 20, "spc": 30},
    "GLOOM": {"hp": 60, "atk": 65, "def": 70, "spd": 40, "spc": 85},
    "GOLBAT": {"hp": 75, "atk": 80, "def": 70, "spd": 90, "spc": 75},
    "GOLDEEN": {"hp": 45, "atk": 67, "def": 60, "spd": 63, "spc": 50},
    "GOLDUCK": {"hp": 80, "atk": 82, "def": 78, "spd": 85, "spc": 80},
    "GOLEM": {"hp": 80, "atk": 110, "def": 130, "spd": 45, "spc": 55},
    "GRAVELER": {"hp": 55, "atk": 95, "def": 115, "spd": 35, "spc": 45},
    "GRIMER": {"hp": 80, "atk": 80, "def": 50, "spd": 25, "spc": 40},
    "GROWLITHE": {"hp": 55, "atk": 70, "def": 45, "spd": 60, "spc": 50},
    "GYARADOS": {"hp": 95, "atk": 125, "def": 79, "spd": 81, "spc": 100},
    "HAUNTER": {"hp": 45, "atk": 50, "def": 45, "spd": 95, "spc": 115},
    "HITMONCHAN": {"hp": 50, "atk": 105, "def": 79, "spd": 76, "spc": 35},
    "HITMONLEE": {"hp": 50, "atk": 120, "def": 53, "spd": 87, "spc": 35},
    "HORSEA": {"hp": 30, "atk": 40, "def": 70, "spd": 60, "spc": 70},
    "HYPNO": {"hp": 85, "atk": 73, "def": 70, "spd": 67, "spc": 115},
    "IVYSAUR": {"hp": 60, "atk": 62, "def": 63, "spd": 60, "spc": 80},
    "JIGGLYPUFF": {"hp": 115, "atk": 45, "def": 20, "spd": 20, "spc": 25},
    "JOLTEON": {"hp": 65, "atk": 65, "def": 60, "spd": 130, "spc": 110},
    "JYNX": {"hp": 65, "atk": 50, "def": 35, "spd": 95, "spc": 95},
    "KABUTO": {"hp": 30, "atk": 80, "def": 90, "spd": 55, "spc": 45},
    "KABUTOPS": {"hp": 60, "atk": 115, "def": 105, "spd": 80, "spc": 70},
    "KADABRA": {"hp": 40, "atk": 35, "def": 30, "spd": 105, "spc": 120},
    "KAKUNA": {"hp": 45, "atk": 25, "def": 50, "spd": 35, "spc": 25},
    "KANGASKHAN": {"hp": 105, "atk": 95, "def": 80, "spd": 90, "spc": 40},
    "KINGLER": {"hp": 55, "atk": 130, "def": 115, "spd": 75, "spc": 50},
    "KOFFING": {"hp": 40, "atk": 65, "def": 95, "spd": 35, "spc": 60},
    "KRABBY": {"hp": 30, "atk": 105, "def": 90, "spd": 50, "spc": 25},
    "LAPRAS": {"hp": 130, "atk": 85, "def": 80, "spd": 60, "spc": 95},
    "LICKITUNG": {"hp": 90, "atk": 55, "def": 75, "spd": 30, "spc": 60},
    "MACHAMP": {"hp": 90, "atk": 130, "def": 80, "spd": 55, "spc": 65},
    "MACHOKE": {"hp": 80, "atk": 100, "def": 70, "spd": 45, "spc": 50},
    "MACHOP": {"hp": 70, "atk": 80, "def": 50, "spd": 35, "spc": 35},
    "MAGIKARP": {"hp": 20, "atk": 10, "def": 55, "spd": 80, "spc": 20},
    "MAGMAR": {"hp": 65, "atk": 95, "def": 57, "spd": 93, "spc": 85},
    "MAGNEMITE": {"hp": 25, "atk": 35, "def": 70, "spd": 45, "spc": 95},
    "MAGNETON": {"hp": 50, "atk": 60, "def": 95, "spd": 70, "spc": 120},
    "MANKEY": {"hp": 40, "atk": 80, "def": 35, "spd": 70, "spc": 35},
    "MAROWAK": {"hp": 60, "atk": 80, "def": 110, "spd": 45, "spc": 50},
    "MEOWTH": {"hp": 40, "atk": 45, "def": 35, "spd": 90, "spc": 40},
    "METAPOD": {"hp": 50, "atk": 20, "def": 55, "spd": 30, "spc": 25},
    "MEW": {"hp": 100, "atk": 100, "def": 100, "spd": 100, "spc": 100},
    "MEWTWO": {"hp": 106, "atk": 110, "def": 90, "spd": 130, "spc": 154},
    "MOLTRES": {"hp": 90, "atk": 100, "def": 90, "spd": 90, "spc": 125},
    "MRMIME": {"hp": 40, "atk": 45, "def": 65, "spd": 90, "spc": 100},
    "MUK": {"hp": 105, "atk": 105, "def": 75, "spd": 50, "spc": 65},
    "NIDOKING": {"hp": 81, "atk": 92, "def": 77, "spd": 85, "spc": 75},
    "NIDOQUEEN": {"hp": 90, "atk": 82, "def": 87, "spd": 76, "spc": 75},
    "NIDORANF": {"hp": 55, "atk": 47, "def": 52, "spd": 41, "spc": 40},
    "NIDORANM": {"hp": 46, "atk": 57, "def": 40, "spd": 50, "spc": 40},
    "NIDORINA": {"hp": 70, "atk": 62, "def": 67, "spd": 56, "spc": 55},
    "NIDORINO": {"hp": 61, "atk": 72, "def": 57, "spd": 65, "spc": 55},
    "NINETALES": {"hp": 73, "atk": 76, "def": 75, "spd": 100, "spc": 100},
    "ODDISH": {"hp": 45, "atk": 50, "def": 55, "spd": 30, "spc": 75},
    "OMANYTE": {"hp": 35, "atk": 40, "def": 100, "spd": 35, "spc": 90},
    "OMASTAR": {"hp": 70, "atk": 60, "def": 125, "spd": 55, "spc": 115},
    "ONIX": {"hp": 35, "atk": 45, "def": 160, "spd": 70, "spc": 30},
    "PARAS": {"hp": 35, "atk": 70, "def": 55, "spd": 25, "spc": 55},
    "PARASECT": {"hp": 60, "atk": 95, "def": 80, "spd": 30, "spc": 80},
    "PERSIAN": {"hp": 65, "atk": 70, "def": 60, "spd": 115, "spc": 65},
    "PIDGEOT": {"hp": 83, "atk": 80, "def": 75, "spd": 91, "spc": 70},
    "PIDGEOTTO": {"hp": 63, "atk": 60, "def": 55, "spd": 71, "spc": 50},
    "PIDGEY": {"hp": 40, "atk": 45, "def": 40, "spd": 56, "spc": 35},
    "PIKACHU": {"hp": 35, "atk": 55, "def": 30, "spd": 90, "spc": 50},
    "PINSIR": {"hp": 65, "atk": 125, "def": 100, "spd": 85, "spc": 55},
    "POLIWAG": {"hp": 40, "atk": 50, "def": 40, "spd": 90, "spc": 40},
    "POLIWHIRL": {"hp": 65, "atk": 65, "def": 65, "spd": 90, "spc": 50},
    "POLIWRATH": {"hp": 90, "atk": 85, "def": 95, "spd": 70, "spc": 70},
    "PONYTA": {"hp": 50, "atk": 85, "def": 55, "spd": 90, "spc": 65},
    "PORYGON": {"hp": 65, "atk": 60, "def": 70, "spd": 40, "spc": 75},
    "PRIMEAPE": {"hp": 65, "atk": 105, "def": 60, "spd": 95, "spc": 60},
    "PSYDUCK": {"hp": 50, "atk": 52, "def": 48, "spd": 55, "spc": 50},
    "RAICHU": {"hp": 60, "atk": 90, "def": 55, "spd": 100, "spc": 90},
    "RAPIDASH": {"hp": 65, "atk": 100, "def": 70, "spd": 105, "spc": 80},
    "RATICATE": {"hp": 55, "atk": 81, "def": 60, "spd": 97, "spc": 50},
    "RATTATA": {"hp": 30, "atk": 56, "def": 35, "spd": 72, "spc": 25},
    "RHYDON": {"hp": 105, "atk": 130, "def": 120, "spd": 40, "spc": 45},
    "RHYHORN": {"hp": 80, "atk": 85, "def": 95, "spd": 25, "spc": 30},
    "SANDSHREW": {"hp": 50, "atk": 75, "def": 85, "spd": 40, "spc": 30},
    "SANDSLASH": {"hp": 75, "atk": 100, "def": 110, "spd": 65, "spc": 55},
    "SCYTHER": {"hp": 70, "atk": 110, "def": 80, "spd": 105, "spc": 55},
    "SEADRA": {"hp": 55, "atk": 65, "def": 95, "spd": 85, "spc": 95},
    "SEAKING": {"hp": 80, "atk": 92, "def": 65, "spd": 68, "spc": 80},
    "SEEL": {"hp": 65, "atk": 45, "def": 55, "spd": 45, "spc": 70},
    "SHELLDER": {"hp": 30, "atk": 65, "def": 100, "spd": 40, "spc": 45},
    "SLOWBRO": {"hp": 95, "atk": 75, "def": 110, "spd": 30, "spc": 80},
    "SLOWPOKE": {"hp": 90, "atk": 65, "def": 65, "spd": 15, "spc": 40},
    "SNORLAX": {"hp": 160, "atk": 110, "def": 65, "spd": 30, "spc": 65},
    "SPEAROW": {"hp": 40, "atk": 60, "def": 30, "spd": 70, "spc": 31},
    "SQUIRTLE": {"hp": 44, "atk": 48, "def": 65, "spd": 43, "spc": 50},
    "STARMIE": {"hp": 60, "atk": 75, "def": 85, "spd": 115, "spc": 100},
    "STARYU": {"hp": 30, "atk": 45, "def": 55, "spd": 85, "spc": 70},
    "TANGELA": {"hp": 65, "atk": 55, "def": 115, "spd": 60, "spc": 100},
    "TAUROS": {"hp": 75, "atk": 100, "def": 95, "spd": 110, "spc": 70},
    "TENTACOOL": {"hp": 40, "atk": 40, "def": 35, "spd": 70, "spc": 100},
    "TENTACRUEL": {"hp": 80, "atk": 70, "def": 65, "spd": 100, "spc": 120},
    "VAPOREON": {"hp": 130, "atk": 65, "def": 60, "spd": 65, "spc": 110},
    "VENOMOTH": {"hp": 70, "atk": 65, "def": 60, "spd": 90, "spc": 90},
    "VENONAT": {"hp": 60, "atk": 55, "def": 50, "spd": 45, "spc": 40},
    "VENUSAUR": {"hp": 80, "atk": 82, "def": 83, "spd": 80, "spc": 100},
    "VICTREEBEL": {"hp": 80, "atk": 105, "def": 65, "spd": 70, "spc": 100},
    "VILEPLUME": {"hp": 75, "atk": 80, "def": 85, "spd": 50, "spc": 100},
    "VOLTORB": {"hp": 40, "atk": 30, "def": 50, "spd": 100, "spc": 55},
    "VULPIX": {"hp": 38, "atk": 41, "def": 40, "spd": 65, "spc": 65},
    "WARTORTLE": {"hp": 59, "atk": 63, "def": 80, "spd": 58, "spc": 65},
    "WEEDLE": {"hp": 40, "atk": 35, "def": 30, "spd": 50, "spc": 20},
    "WEEPINBELL": {"hp": 65, "atk": 90, "def": 50, "spd": 55, "spc": 85},
    "WEEZING": {"hp": 65, "atk": 90, "def": 120, "spd": 60, "spc": 85},
    "WIGGLYTUFF": {"hp": 140, "atk": 70, "def": 45, "spd": 45, "spc": 50},
    "ZAPDOS": {"hp": 90, "atk": 90, "def": 85, "spd": 100, "spc": 125},
    "ZUBAT": {"hp": 40, "atk": 45, "def": 35, "spd": 55, "spc": 40},
    "NIDORAN_M": {"hp": 46, "atk": 57, "def": 40, "spd": 50, "spc": 40},
    "NIDORAN_F": {"hp": 55, "atk": 47, "def": 52, "spd": 41, "spc": 40},
    "MR_MIME": {"hp": 40, "atk": 45, "def": 65, "spd": 90, "spc": 100},
}

SPECIES_TO_ID = {
    "NO_MON": 0,
    "RHYDON": 1,
    "KANGASKHAN": 2,
    "NIDORAN_M": 3,
    "CLEFAIRY": 4,
    "SPEAROW": 5,
    "VOLTORB": 6,
    "NIDOKING": 7,
    "SLOWBRO": 8,
    "IVYSAUR": 9,
    "EXEGGUTOR": 10,
    "LICKITUNG": 11,
    "EXEGGCUTE": 12,
    "GRIMER": 13,
    "GENGAR": 14,
    "NIDORAN_F": 15,
    "NIDOQUEEN": 16,
    "CUBONE": 17,
    "RHYHORN": 18,
    "LAPRAS": 19,
    "ARCANINE": 20,
    "MEW": 21,
    "GYARADOS": 22,
    "SHELLDER": 23,
    "TENTACOOL": 24,
    "GASTLY": 25,
    "SCYTHER": 26,
    "STARYU": 27,
    "BLASTOISE": 28,
    "PINSIR": 29,
    "TANGELA": 30,
    "GROWLITHE": 33,
    "ONIX": 34,
    "FEAROW": 35,
    "PIDGEY": 36,
    "SLOWPOKE": 37,
    "KADABRA": 38,
    "GRAVELER": 39,
    "CHANSEY": 40,
    "MACHOKE": 41,
    "MR_MIME": 42,
    "HITMONLEE": 43,
    "HITMONCHAN": 44,
    "ARBOK": 45,
    "PARASECT": 46,
    "PSYDUCK": 47,
    "DROWZEE": 48,
    "GOLEM": 49,
    "MAGMAR": 51,
    "ELECTABUZZ": 53,
    "MAGNETON": 54,
    "KOFFING": 55,
    "MANKEY": 57,
    "SEEL": 58,
    "DIGLETT": 59,
    "TAUROS": 60,
    "FARFETCHD": 64,
    "VENONAT": 65,
    "DRAGONITE": 66,
    "DODUO": 70,
    "POLIWAG": 71,
    "JYNX": 72,
    "MOLTRES": 73,
    "ARTICUNO": 74,
    "ZAPDOS": 75,
    "DITTO": 76,
    "MEOWTH": 77,
    "KRABBY": 78,
    "VULPIX": 82,
    "NINETALES": 83,
    "PIKACHU": 84,
    "RAICHU": 85,
    "DRATINI": 88,
    "DRAGONAIR": 89,
    "KABUTO": 90,
    "KABUTOPS": 91,
    "HORSEA": 92,
    "SEADRA": 93,
    "SANDSHREW": 96,
    "SANDSLASH": 97,
    "OMANYTE": 98,
    "OMASTAR": 99,
    "JIGGLYPUFF": 100,
    "WIGGLYTUFF": 101,
    "EEVEE": 102,
    "FLAREON": 103,
    "JOLTEON": 104,
    "VAPOREON": 105,
    "MACHOP": 106,
    "ZUBAT": 107,
    "EKANS": 108,
    "PARAS": 109,
    "POLIWHIRL": 110,
    "POLIWRATH": 111,
    "WEEDLE": 112,
    "KAKUNA": 113,
    "BEEDRILL": 114,
    "DODRIO": 116,
    "PRIMEAPE": 117,
    "DUGTRIO": 118,
    "VENOMOTH": 119,
    "DEWGONG": 120,
    "CATERPIE": 123,
    "METAPOD": 124,
    "BUTTERFREE": 125,
    "MACHAMP": 126,
    "GOLDUCK": 128,
    "HYPNO": 129,
    "GOLBAT": 130,
    "MEWTWO": 131,
    "SNORLAX": 132,
    "MAGIKARP": 133,
    "MUK": 136,
    "KINGLER": 138,
    "CLOYSTER": 139,
    "ELECTRODE": 141,
    "CLEFABLE": 142,
    "WEEZING": 143,
    "PERSIAN": 144,
    "MAROWAK": 145,
    "HAUNTER": 147,
    "ABRA": 148,
    "ALAKAZAM": 149,
    "PIDGEOTTO": 150,
    "PIDGEOT": 151,
    "STARMIE": 152,
    "BULBASAUR": 153,
    "VENUSAUR": 154,
    "TENTACRUEL": 155,
    "GOLDEEN": 157,
    "SEAKING": 158,
    "PONYTA": 163,
    "RAPIDASH": 164,
    "RATTATA": 165,
    "RATICATE": 166,
    "NIDORINO": 167,
    "NIDORINA": 168,
    "GEODUDE": 169,
    "PORYGON": 170,
    "AERODACTYL": 171,
    "MAGNEMITE": 173,
    "CHARMANDER": 176,
    "SQUIRTLE": 177,
    "CHARMELEON": 178,
    "WARTORTLE": 179,
    "CHARIZARD": 180,
    "FOSSIL_KABUTOPS": 182,
    "FOSSIL_AERODACTYL": 183,
    "MON_GHOST": 184,
    "ODDISH": 185,
    "GLOOM": 186,
    "VILEPLUME": 187,
    "BELLSPROUT": 188,
    "WEEPINBELL": 189,
    "VICTREEBEL": 190,
}

ID_TO_SPECIES = {
    0: "NO_MON",
    1: "RHYDON",
    2: "KANGASKHAN",
    3: "NIDORAN_M",
    4: "CLEFAIRY",
    5: "SPEAROW",
    6: "VOLTORB",
    7: "NIDOKING",
    8: "SLOWBRO",
    9: "IVYSAUR",
    10: "EXEGGUTOR",
    11: "LICKITUNG",
    12: "EXEGGCUTE",
    13: "GRIMER",
    14: "GENGAR",
    15: "NIDORAN_F",
    16: "NIDOQUEEN",
    17: "CUBONE",
    18: "RHYHORN",
    19: "LAPRAS",
    20: "ARCANINE",
    21: "MEW",
    22: "GYARADOS",
    23: "SHELLDER",
    24: "TENTACOOL",
    25: "GASTLY",
    26: "SCYTHER",
    27: "STARYU",
    28: "BLASTOISE",
    29: "PINSIR",
    30: "TANGELA",
    33: "GROWLITHE",
    34: "ONIX",
    35: "FEAROW",
    36: "PIDGEY",
    37: "SLOWPOKE",
    38: "KADABRA",
    39: "GRAVELER",
    40: "CHANSEY",
    41: "MACHOKE",
    42: "MR_MIME",
    43: "HITMONLEE",
    44: "HITMONCHAN",
    45: "ARBOK",
    46: "PARASECT",
    47: "PSYDUCK",
    48: "DROWZEE",
    49: "GOLEM",
    51: "MAGMAR",
    53: "ELECTABUZZ",
    54: "MAGNETON",
    55: "KOFFING",
    57: "MANKEY",
    58: "SEEL",
    59: "DIGLETT",
    60: "TAUROS",
    64: "FARFETCHD",
    65: "VENONAT",
    66: "DRAGONITE",
    70: "DODUO",
    71: "POLIWAG",
    72: "JYNX",
    73: "MOLTRES",
    74: "ARTICUNO",
    75: "ZAPDOS",
    76: "DITTO",
    77: "MEOWTH",
    78: "KRABBY",
    82: "VULPIX",
    83: "NINETALES",
    84: "PIKACHU",
    85: "RAICHU",
    88: "DRATINI",
    89: "DRAGONAIR",
    90: "KABUTO",
    91: "KABUTOPS",
    92: "HORSEA",
    93: "SEADRA",
    96: "SANDSHREW",
    97: "SANDSLASH",
    98: "OMANYTE",
    99: "OMASTAR",
    100: "JIGGLYPUFF",
    101: "WIGGLYTUFF",
    102: "EEVEE",
    103: "FLAREON",
    104: "JOLTEON",
    105: "VAPOREON",
    106: "MACHOP",
    107: "ZUBAT",
    108: "EKANS",
    109: "PARAS",
    110: "POLIWHIRL",
    111: "POLIWRATH",
    112: "WEEDLE",
    113: "KAKUNA",
    114: "BEEDRILL",
    116: "DODRIO",
    117: "PRIMEAPE",
    118: "DUGTRIO",
    119: "VENOMOTH",
    120: "DEWGONG",
    123: "CATERPIE",
    124: "METAPOD",
    125: "BUTTERFREE",
    126: "MACHAMP",
    128: "GOLDUCK",
    129: "HYPNO",
    130: "GOLBAT",
    131: "MEWTWO",
    132: "SNORLAX",
    133: "MAGIKARP",
    136: "MUK",
    138: "KINGLER",
    139: "CLOYSTER",
    141: "ELECTRODE",
    142: "CLEFABLE",
    143: "WEEZING",
    144: "PERSIAN",
    145: "MAROWAK",
    147: "HAUNTER",
    148: "ABRA",
    149: "ALAKAZAM",
    150: "PIDGEOTTO",
    151: "PIDGEOT",
    152: "STARMIE",
    153: "BULBASAUR",
    154: "VENUSAUR",
    155: "TENTACRUEL",
    157: "GOLDEEN",
    158: "SEAKING",
    163: "PONYTA",
    164: "RAPIDASH",
    165: "RATTATA",
    166: "RATICATE",
    167: "NIDORINO",
    168: "NIDORINA",
    169: "GEODUDE",
    170: "PORYGON",
    171: "AERODACTYL",
    173: "MAGNEMITE",
    176: "CHARMANDER",
    177: "SQUIRTLE",
    178: "CHARMELEON",
    179: "WARTORTLE",
    180: "CHARIZARD",
    182: "FOSSIL_KABUTOPS",
    183: "FOSSIL_AERODACTYL",
    184: "MON_GHOST",
    185: "ODDISH",
    186: "GLOOM",
    187: "VILEPLUME",
    188: "BELLSPROUT",
    189: "WEEPINBELL",
    190: "VICTREEBEL",
}

CHARMAP = {
    "<NULL>": 0,
    "<PAGE>": 73,
    "<PKMN>": 74,
    "<_CONT>": 75,
    "<SCROLL>": 76,
    "<NEXT>": 78,
    "<LINE>": 79,
    "@": 80,
    "<PARA>": 81,
    "<PLAYER>": 82,
    "<RIVAL>": 83,
    "#": 84,
    "<CONT>": 85,
    "<……>": 86,
    "<DONE>": 87,
    "<PROMPT>": 88,
    "<TARGET>": 89,
    "<USER>": 90,
    "<PC>": 91,
    "<TM>": 92,
    "<TRAINER>": 93,
    "<ROCKET>": 94,
    "<DEXEND>": 95,
    "<BOLD_A>": 96,
    "<BOLD_B>": 97,
    "<BOLD_C>": 98,
    "<BOLD_D>": 99,
    "<BOLD_E>": 100,
    "<BOLD_F>": 101,
    "<BOLD_G>": 102,
    "<BOLD_H>": 103,
    "<BOLD_I>": 104,
    "<BOLD_V>": 105,
    "<BOLD_S>": 106,
    "<BOLD_L>": 107,
    "<BOLD_M>": 108,
    "<COLON>": 109,
    "ぃ": 110,
    "ぅ": 111,
    "‘": 112,
    "’": 113,
    "“": 114,
    "”": 115,
    "·": 116,
    "…": 117,
    "ぁ": 118,
    "ぇ": 119,
    "ぉ": 120,
    "┌": 121,
    "─": 122,
    "┐": 123,
    "│": 124,
    "└": 125,
    "┘": 126,
    "<LV>": 110,
    "<to>": 112,
    "『": 114,
    "<ID>": 115,
    "№": 116,
    "′": 96,
    "″": 97,
    "<BOLD_P>": 114,
    "▲": 237,
    "<ED>": 240,
    "A": 128,
    "B": 129,
    "C": 130,
    "D": 131,
    "E": 132,
    "F": 133,
    "G": 134,
    "H": 135,
    "I": 136,
    "J": 137,
    "K": 138,
    "L": 139,
    "M": 140,
    "N": 141,
    "O": 142,
    "P": 143,
    "Q": 144,
    "R": 145,
    "S": 146,
    "T": 147,
    "U": 148,
    "V": 149,
    "W": 150,
    "X": 151,
    "Y": 152,
    "Z": 153,
    "(": 154,
    ")": 155,
    ":": 156,
    ";": 157,
    "[": 158,
    "]": 159,
    "a": 160,
    "b": 161,
    "c": 162,
    "d": 163,
    "e": 164,
    "f": 165,
    "g": 166,
    "h": 167,
    "i": 168,
    "j": 169,
    "k": 170,
    "l": 171,
    "m": 172,
    "n": 173,
    "o": 174,
    "p": 175,
    "q": 176,
    "r": 177,
    "s": 178,
    "t": 179,
    "u": 180,
    "v": 181,
    "w": 182,
    "x": 183,
    "y": 184,
    "z": 185,
    "é": 186,
    "'d": 187,
    "'l": 188,
    "'s": 189,
    "'t": 190,
    "'v": 191,
    "'": 224,
    "<PK>": 225,
    "<MN>": 226,
    "-": 227,
    "'r": 228,
    "'m": 229,
    "?": 230,
    "!": 231,
    ".": 232,
    "ァ": 233,
    "ゥ": 234,
    "ェ": 235,
    "▷": 236,
    "▶": 237,
    "▼": 238,
    "♂": 239,
    "¥": 240,
    "×": 241,
    "<DOT>": 242,
    "/": 243,
    ",": 244,
    "♀": 245,
    "0": 246,
    "1": 247,
    "2": 248,
    "3": 249,
    "4": 250,
    "5": 251,
    "6": 252,
    "7": 253,
    "8": 254,
    "9": 255,
}

MOVES_INFO_DICT = {
    1: {
        "name": "POUND",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 40,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 35,
        "power": 40.0,
    },
    2: {
        "name": "KARATE_CHOP",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 50,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 25,
        "power": 50.0,
    },
    3: {
        "name": "DOUBLESLAP",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 15,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 10,
        "power": 37.8984375,
    },
    4: {
        "name": "COMET_PUNCH",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 18,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 15,
        "power": 48.7265625,
    },
    5: {
        "name": "MEGA_PUNCH",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 77.0,
    },
    6: {
        "name": "PAY_DAY",
        "effect": "PAY_DAY_EFFECT",
        "raw_power": 40,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 60.0,
    },
    7: {
        "name": "FIRE_PUNCH",
        "effect": "BURN_SIDE_EFFECT1",
        "raw_power": 75,
        "type": "FIRE",
        "type_id": 20,
        "accuracy": 100,
        "pp": 15,
        "power": 79.6875,
    },
    8: {
        "name": "ICE_PUNCH",
        "effect": "FREEZE_SIDE_EFFECT",
        "raw_power": 75,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 100,
        "pp": 15,
        "power": 89.0625,
    },
    9: {
        "name": "THUNDERPUNCH",
        "effect": "PARALYZE_SIDE_EFFECT1",
        "raw_power": 75,
        "type": "ELECTRIC",
        "type_id": 23,
        "accuracy": 100,
        "pp": 15,
        "power": 89.0625,
    },
    10: {
        "name": "SCRATCH",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 40,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 35,
        "power": 40.0,
    },
    11: {
        "name": "VICEGRIP",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 55,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 55.0,
    },
    12: {
        "name": "GUILLOTINE",
        "effect": "OHKO_EFFECT",
        "raw_power": 1,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 30,
        "pp": 5,
        "power": 40.8890625,
    },
    13: {
        "name": "RAZOR_WIND",
        "effect": "CHARGE_EFFECT",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 10,
        "power": 52.5,
    },
    14: {
        "name": "SWORDS_DANCE",
        "effect": "ATTACK_UP2_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 20.0,
    },
    15: {
        "name": "CUT",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 50,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 95,
        "pp": 30,
        "power": 49.375,
    },
    16: {
        "name": "GUST",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 40,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 35,
        "power": 40.0,
    },
    17: {
        "name": "WING_ATTACK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 35,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 100,
        "pp": 35,
        "power": 35.0,
    },
    18: {
        "name": "WHIRLWIND",
        "effect": "SWITCH_AND_TELEPORT_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 0.0,
    },
    19: {
        "name": "FLY",
        "effect": "FLY_EFFECT",
        "raw_power": 70,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 95,
        "pp": 15,
        "power": 64.8046875,
    },
    20: {
        "name": "BIND",
        "effect": "TRAPPING_EFFECT",
        "raw_power": 15,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 20,
        "power": 56.25,
    },
    21: {
        "name": "SLAM",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 20,
        "power": 75.0,
    },
    22: {
        "name": "VINE_WHIP",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 35,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 10,
        "power": 30.625,
    },
    23: {
        "name": "STOMP",
        "effect": "FLINCH_SIDE_EFFECT2",
        "raw_power": 65,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 105.0,
    },
    24: {
        "name": "DOUBLE_KICK",
        "effect": "ATTACK_TWICE_EFFECT",
        "raw_power": 30,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 100,
        "pp": 30,
        "power": 60.0,
    },
    25: {
        "name": "MEGA_KICK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 120,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 5,
        "power": 91.40625,
    },
    26: {
        "name": "JUMP_KICK",
        "effect": "JUMP_KICK_EFFECT",
        "raw_power": 70,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 95,
        "pp": 25,
        "power": 62.212500000000006,
    },
    27: {
        "name": "ROLLING_KICK",
        "effect": "FLINCH_SIDE_EFFECT2",
        "raw_power": 60,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 85,
        "pp": 15,
        "power": 90.234375,
    },
    28: {
        "name": "SAND_ATTACK",
        "effect": "ACCURACY_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 9.375,
    },
    29: {
        "name": "HEADBUTT",
        "effect": "FLINCH_SIDE_EFFECT2",
        "raw_power": 70,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 103.125,
    },
    30: {
        "name": "HORN_ATTACK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 65,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 25,
        "power": 65.0,
    },
    31: {
        "name": "FURY_ATTACK",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 15,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 43.3125,
    },
    32: {
        "name": "HORN_DRILL",
        "effect": "OHKO_EFFECT",
        "raw_power": 1,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 30,
        "pp": 5,
        "power": 40.8890625,
    },
    33: {
        "name": "TACKLE",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 35,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 95,
        "pp": 35,
        "power": 34.5625,
    },
    34: {
        "name": "BODY_SLAM",
        "effect": "PARALYZE_SIDE_EFFECT2",
        "raw_power": 85,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 98.4375,
    },
    35: {
        "name": "WRAP",
        "effect": "TRAPPING_EFFECT",
        "raw_power": 15,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 57.75,
    },
    36: {
        "name": "TAKE_DOWN",
        "effect": "RECOIL_EFFECT",
        "raw_power": 90,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 77.9625,
    },
    37: {
        "name": "THRASH",
        "effect": "THRASH_PETAL_DANCE_EFFECT",
        "raw_power": 90,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 81.0,
    },
    38: {
        "name": "DOUBLE_EDGE",
        "effect": "RECOIL_EFFECT",
        "raw_power": 100,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 84.375,
    },
    39: {
        "name": "TAIL_WHIP",
        "effect": "DEFENSE_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    40: {
        "name": "POISON_STING",
        "effect": "POISON_SIDE_EFFECT1",
        "raw_power": 15,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 100,
        "pp": 35,
        "power": 25.0,
    },
    41: {
        "name": "TWINEEDLE",
        "effect": "TWINEEDLE_EFFECT",
        "raw_power": 25,
        "type": "BUG",
        "type_id": 7,
        "accuracy": 100,
        "pp": 20,
        "power": 62.0,
    },
    42: {
        "name": "PIN_MISSILE",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 14,
        "type": "BUG",
        "type_id": 7,
        "accuracy": 85,
        "pp": 20,
        "power": 40.425000000000004,
    },
    43: {
        "name": "LEER",
        "effect": "DEFENSE_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    44: {
        "name": "BITE",
        "effect": "FLINCH_SIDE_EFFECT1",
        "raw_power": 60,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 25,
        "power": 80.0,
    },
    45: {
        "name": "GROWL",
        "effect": "ATTACK_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 40,
        "power": 10.0,
    },
    46: {
        "name": "ROAR",
        "effect": "SWITCH_AND_TELEPORT_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 0.0,
    },
    47: {
        "name": "SING",
        "effect": "SLEEP_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 55,
        "pp": 15,
        "power": 16.640625,
    },
    48: {
        "name": "SUPERSONIC",
        "effect": "CONFUSION_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 55,
        "pp": 20,
        "power": 8.875,
    },
    49: {
        "name": "SONICBOOM",
        "effect": "SPECIAL_DAMAGE_EFFECT",
        "raw_power": 1,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 90,
        "pp": 20,
        "power": 39.975,
    },
    50: {
        "name": "DISABLE",
        "effect": "DISABLE_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 55,
        "pp": 20,
        "power": 17.75,
    },
    51: {
        "name": "ACID",
        "effect": "DEFENSE_DOWN_SIDE_EFFECT",
        "raw_power": 40,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 100,
        "pp": 30,
        "power": 50.0,
    },
    52: {
        "name": "EMBER",
        "effect": "BURN_SIDE_EFFECT1",
        "raw_power": 40,
        "type": "FIRE",
        "type_id": 20,
        "accuracy": 100,
        "pp": 25,
        "power": 50.0,
    },
    53: {
        "name": "FLAMETHROWER",
        "effect": "BURN_SIDE_EFFECT1",
        "raw_power": 95,
        "type": "FIRE",
        "type_id": 20,
        "accuracy": 100,
        "pp": 15,
        "power": 98.4375,
    },
    54: {
        "name": "MIST",
        "effect": "MIST_EFFECT",
        "raw_power": 0,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    55: {
        "name": "WATER_GUN",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 40,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 25,
        "power": 40.0,
    },
    56: {
        "name": "HYDRO_PUMP",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 120,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 80,
        "pp": 5,
        "power": 92.625,
    },
    57: {
        "name": "SURF",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 95,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 15,
        "power": 89.0625,
    },
    58: {
        "name": "ICE_BEAM",
        "effect": "FREEZE_SIDE_EFFECT",
        "raw_power": 95,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 100,
        "pp": 10,
        "power": 100.625,
    },
    59: {
        "name": "BLIZZARD",
        "effect": "FREEZE_SIDE_EFFECT",
        "raw_power": 120,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 90,
        "pp": 5,
        "power": 110.90624999999999,
    },
    60: {
        "name": "PSYBEAM",
        "effect": "CONFUSION_SIDE_EFFECT",
        "raw_power": 65,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 20,
        "power": 95.0,
    },
    61: {
        "name": "BUBBLEBEAM",
        "effect": "SPEED_DOWN_SIDE_EFFECT",
        "raw_power": 65,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 20,
        "power": 75.0,
    },
    62: {
        "name": "AURORA_BEAM",
        "effect": "ATTACK_DOWN_SIDE_EFFECT",
        "raw_power": 65,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 100,
        "pp": 20,
        "power": 75.0,
    },
    63: {
        "name": "HYPER_BEAM",
        "effect": "HYPER_BEAM_EFFECT",
        "raw_power": 150,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 90,
        "pp": 5,
        "power": 88.725,
    },
    64: {
        "name": "PECK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 35,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 100,
        "pp": 35,
        "power": 35.0,
    },
    65: {
        "name": "DRILL_PECK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 100,
        "pp": 20,
        "power": 80.0,
    },
    66: {
        "name": "SUBMISSION",
        "effect": "RECOIL_EFFECT",
        "raw_power": 80,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 80,
        "pp": 25,
        "power": 68.39999999999999,
    },
    67: {
        "name": "LOW_KICK",
        "effect": "FLINCH_SIDE_EFFECT2",
        "raw_power": 50,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 90,
        "pp": 20,
        "power": 87.75,
    },
    68: {
        "name": "COUNTER",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 1,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 100,
        "pp": 20,
        "power": 1.0,
    },
    69: {
        "name": "SEISMIC_TOSS",
        "effect": "SPECIAL_DAMAGE_EFFECT",
        "raw_power": 1,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 100,
        "pp": 20,
        "power": 41.0,
    },
    70: {
        "name": "STRENGTH",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 75.0,
    },
    71: {
        "name": "ABSORB",
        "effect": "DRAIN_HP_EFFECT",
        "raw_power": 20,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 20,
        "power": 30.0,
    },
    72: {
        "name": "MEGA_DRAIN",
        "effect": "DRAIN_HP_EFFECT",
        "raw_power": 40,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 10,
        "power": 52.5,
    },
    73: {
        "name": "LEECH_SEED",
        "effect": "LEECH_SEED_EFFECT",
        "raw_power": 0,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 90,
        "pp": 10,
        "power": 34.125,
    },
    74: {
        "name": "GROWTH",
        "effect": "SPECIAL_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 40,
        "power": 10.0,
    },
    75: {
        "name": "RAZOR_LEAF",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 55,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 95,
        "pp": 25,
        "power": 54.3125,
    },
    76: {
        "name": "SOLARBEAM",
        "effect": "CHARGE_EFFECT",
        "raw_power": 120,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 10,
        "power": 84.0,
    },
    77: {
        "name": "POISONPOWDER",
        "effect": "POISON_EFFECT",
        "raw_power": 0,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 75,
        "pp": 35,
        "power": 18.75,
    },
    78: {
        "name": "STUN_SPORE",
        "effect": "PARALYZE_EFFECT",
        "raw_power": 0,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 75,
        "pp": 30,
        "power": 28.125,
    },
    79: {
        "name": "SLEEP_POWDER",
        "effect": "SLEEP_EFFECT",
        "raw_power": 0,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 75,
        "pp": 15,
        "power": 17.578125,
    },
    80: {
        "name": "PETAL_DANCE",
        "effect": "THRASH_PETAL_DANCE_EFFECT",
        "raw_power": 70,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 20,
        "power": 63.0,
    },
    81: {
        "name": "STRING_SHOT",
        "effect": "SPEED_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "BUG",
        "type_id": 7,
        "accuracy": 95,
        "pp": 40,
        "power": 9.875,
    },
    82: {
        "name": "DRAGON_RAGE",
        "effect": "SPECIAL_DAMAGE_EFFECT",
        "raw_power": 1,
        "type": "DRAGON",
        "type_id": 26,
        "accuracy": 100,
        "pp": 10,
        "power": 35.875,
    },
    83: {
        "name": "FIRE_SPIN",
        "effect": "TRAPPING_EFFECT",
        "raw_power": 15,
        "type": "FIRE",
        "type_id": 20,
        "accuracy": 70,
        "pp": 15,
        "power": 52.03125,
    },
    84: {
        "name": "THUNDERSHOCK",
        "effect": "PARALYZE_SIDE_EFFECT1",
        "raw_power": 40,
        "type": "ELECTRIC",
        "type_id": 23,
        "accuracy": 100,
        "pp": 30,
        "power": 60.0,
    },
    85: {
        "name": "THUNDERBOLT",
        "effect": "PARALYZE_SIDE_EFFECT1",
        "raw_power": 95,
        "type": "ELECTRIC",
        "type_id": 23,
        "accuracy": 100,
        "pp": 15,
        "power": 107.8125,
    },
    86: {
        "name": "THUNDER_WAVE",
        "effect": "PARALYZE_EFFECT",
        "raw_power": 0,
        "type": "ELECTRIC",
        "type_id": 23,
        "accuracy": 100,
        "pp": 20,
        "power": 30.0,
    },
    87: {
        "name": "THUNDER",
        "effect": "PARALYZE_SIDE_EFFECT1",
        "raw_power": 120,
        "type": "ELECTRIC",
        "type_id": 23,
        "accuracy": 70,
        "pp": 10,
        "power": 113.31250000000001,
    },
    88: {
        "name": "ROCK_THROW",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 50,
        "type": "ROCK",
        "type_id": 5,
        "accuracy": 65,
        "pp": 15,
        "power": 42.7734375,
    },
    89: {
        "name": "EARTHQUAKE",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 100,
        "type": "GROUND",
        "type_id": 4,
        "accuracy": 100,
        "pp": 10,
        "power": 87.5,
    },
    90: {
        "name": "FISSURE",
        "effect": "OHKO_EFFECT",
        "raw_power": 1,
        "type": "GROUND",
        "type_id": 4,
        "accuracy": 30,
        "pp": 5,
        "power": 40.8890625,
    },
    91: {
        "name": "DIG",
        "effect": "CHARGE_EFFECT",
        "raw_power": 100,
        "type": "GROUND",
        "type_id": 4,
        "accuracy": 100,
        "pp": 10,
        "power": 70.0,
    },
    92: {
        "name": "TOXIC",
        "effect": "POISON_EFFECT",
        "raw_power": 0,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 85,
        "pp": 10,
        "power": 16.84375,
    },
    93: {
        "name": "CONFUSION",
        "effect": "CONFUSION_SIDE_EFFECT",
        "raw_power": 50,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 25,
        "power": 80.0,
    },
    94: {
        "name": "PSYCHIC_M",
        "effect": "SPECIAL_DOWN_SIDE_EFFECT",
        "raw_power": 90,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 10,
        "power": 87.5,
    },
    95: {
        "name": "HYPNOSIS",
        "effect": "SLEEP_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 60,
        "pp": 20,
        "power": 18.0,
    },
    96: {
        "name": "MEDITATE",
        "effect": "ATTACK_UP1_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 40,
        "power": 10.0,
    },
    97: {
        "name": "AGILITY",
        "effect": "SPEED_UP2_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 30,
        "power": 30.0,
    },
    98: {
        "name": "QUICK_ATTACK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 40,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 40.0,
    },
    99: {
        "name": "RAGE",
        "effect": "RAGE_EFFECT",
        "raw_power": 20,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 40.0,
    },
    100: {
        "name": "TELEPORT",
        "effect": "SWITCH_AND_TELEPORT_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 20,
        "power": 0.0,
    },
    101: {
        "name": "NIGHT_SHADE",
        "effect": "SPECIAL_DAMAGE_EFFECT",
        "raw_power": 0,
        "type": "GHOST",
        "type_id": 8,
        "accuracy": 100,
        "pp": 15,
        "power": 37.5,
    },
    102: {
        "name": "MIMIC",
        "effect": "MIMIC_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 8.75,
    },
    103: {
        "name": "SCREECH",
        "effect": "DEFENSE_DOWN2_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 40,
        "power": 19.25,
    },
    104: {
        "name": "DOUBLE_TEAM",
        "effect": "EVASION_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 18.75,
    },
    105: {
        "name": "RECOVER",
        "effect": "HEAL_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 20.0,
    },
    106: {
        "name": "HARDEN",
        "effect": "DEFENSE_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    107: {
        "name": "MINIMIZE",
        "effect": "EVASION_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 20.0,
    },
    108: {
        "name": "SMOKESCREEN",
        "effect": "ACCURACY_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 10.0,
    },
    109: {
        "name": "CONFUSE_RAY",
        "effect": "CONFUSION_EFFECT",
        "raw_power": 0,
        "type": "GHOST",
        "type_id": 8,
        "accuracy": 100,
        "pp": 10,
        "power": 8.75,
    },
    110: {
        "name": "WITHDRAW",
        "effect": "DEFENSE_UP1_EFFECT",
        "raw_power": 0,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 40,
        "power": 10.0,
    },
    111: {
        "name": "DEFENSE_CURL",
        "effect": "DEFENSE_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 40,
        "power": 10.0,
    },
    112: {
        "name": "BARRIER",
        "effect": "DEFENSE_UP2_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 30,
        "power": 20.0,
    },
    113: {
        "name": "LIGHT_SCREEN",
        "effect": "LIGHT_SCREEN_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 30,
        "power": 20.0,
    },
    114: {
        "name": "HAZE",
        "effect": "HAZE_EFFECT",
        "raw_power": 0,
        "type": "ICE",
        "type_id": 25,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    115: {
        "name": "REFLECT",
        "effect": "REFLECT_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 20,
        "power": 20.0,
    },
    116: {
        "name": "FOCUS_ENERGY",
        "effect": "FOCUS_ENERGY_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 20.0,
    },
    117: {
        "name": "BIDE",
        "effect": "BIDE_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 35.0,
    },
    118: {
        "name": "METRONOME",
        "effect": "METRONOME_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 52.5,
    },
    119: {
        "name": "MIRROR_MOVE",
        "effect": "MIRROR_MOVE_EFFECT",
        "raw_power": 0,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 100,
        "pp": 20,
        "power": 30.0,
    },
    120: {
        "name": "SELFDESTRUCT",
        "effect": "EXPLODE_EFFECT",
        "raw_power": 130,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 5,
        "power": 34.125,
    },
    121: {
        "name": "EGG_BOMB",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 100,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 10,
        "power": 82.03125,
    },
    122: {
        "name": "LICK",
        "effect": "PARALYZE_SIDE_EFFECT2",
        "raw_power": 20,
        "type": "GHOST",
        "type_id": 8,
        "accuracy": 100,
        "pp": 30,
        "power": 40.0,
    },
    123: {
        "name": "SMOG",
        "effect": "POISON_SIDE_EFFECT2",
        "raw_power": 20,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 70,
        "pp": 20,
        "power": 46.25,
    },
    124: {
        "name": "SLUDGE",
        "effect": "POISON_SIDE_EFFECT2",
        "raw_power": 65,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 100,
        "pp": 20,
        "power": 95.0,
    },
    125: {
        "name": "BONE_CLUB",
        "effect": "FLINCH_SIDE_EFFECT1",
        "raw_power": 65,
        "type": "GROUND",
        "type_id": 4,
        "accuracy": 85,
        "pp": 20,
        "power": 81.8125,
    },
    126: {
        "name": "FIRE_BLAST",
        "effect": "BURN_SIDE_EFFECT2",
        "raw_power": 120,
        "type": "FIRE",
        "type_id": 20,
        "accuracy": 85,
        "pp": 5,
        "power": 101.66406250000001,
    },
    127: {
        "name": "WATERFALL",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 15,
        "power": 75.0,
    },
    128: {
        "name": "CLAMP",
        "effect": "TRAPPING_EFFECT",
        "raw_power": 35,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 75,
        "pp": 10,
        "power": 114.84375,
    },
    129: {
        "name": "SWIFT",
        "effect": "SWIFT_EFFECT",
        "raw_power": 60,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 80.0,
    },
    130: {
        "name": "SKULL_BASH",
        "effect": "CHARGE_EFFECT",
        "raw_power": 100,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 75.0,
    },
    131: {
        "name": "SPIKE_CANNON",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 20,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 15,
        "power": 56.25,
    },
    132: {
        "name": "CONSTRICT",
        "effect": "SPEED_DOWN_SIDE_EFFECT",
        "raw_power": 10,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 35,
        "power": 20.0,
    },
    133: {
        "name": "AMNESIA",
        "effect": "SPECIAL_UP2_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 20,
        "power": 20.0,
    },
    134: {
        "name": "KINESIS",
        "effect": "ACCURACY_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 80,
        "pp": 15,
        "power": 8.90625,
    },
    135: {
        "name": "SOFTBOILED",
        "effect": "HEAL_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 17.5,
    },
    136: {
        "name": "HI_JUMP_KICK",
        "effect": "JUMP_KICK_EFFECT",
        "raw_power": 85,
        "type": "FIGHTING",
        "type_id": 1,
        "accuracy": 90,
        "pp": 20,
        "power": 74.1,
    },
    137: {
        "name": "GLARE",
        "effect": "PARALYZE_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 30,
        "power": 28.125,
    },
    138: {
        "name": "DREAM_EATER",
        "effect": "DREAM_EATER_EFFECT",
        "raw_power": 100,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 15,
        "power": 30.9375,
    },
    139: {
        "name": "POISON_GAS",
        "effect": "POISON_EFFECT",
        "raw_power": 0,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 55,
        "pp": 40,
        "power": 17.75,
    },
    140: {
        "name": "BARRAGE",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 15,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 85,
        "pp": 20,
        "power": 43.3125,
    },
    141: {
        "name": "LEECH_LIFE",
        "effect": "DRAIN_HP_EFFECT",
        "raw_power": 20,
        "type": "BUG",
        "type_id": 7,
        "accuracy": 100,
        "pp": 15,
        "power": 28.125,
    },
    142: {
        "name": "LOVELY_KISS",
        "effect": "SLEEP_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 75,
        "pp": 10,
        "power": 16.40625,
    },
    143: {
        "name": "SKY_ATTACK",
        "effect": "CHARGE_EFFECT",
        "raw_power": 140,
        "type": "FLYING",
        "type_id": 2,
        "accuracy": 90,
        "pp": 5,
        "power": 88.725,
    },
    144: {
        "name": "TRANSFORM",
        "effect": "TRANSFORM_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 8.75,
    },
    145: {
        "name": "BUBBLE",
        "effect": "SPEED_DOWN_SIDE_EFFECT",
        "raw_power": 20,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 100,
        "pp": 30,
        "power": 30.0,
    },
    146: {
        "name": "DIZZY_PUNCH",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 70,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 61.25,
    },
    147: {
        "name": "SPORE",
        "effect": "SLEEP_EFFECT",
        "raw_power": 0,
        "type": "GRASS",
        "type_id": 22,
        "accuracy": 100,
        "pp": 15,
        "power": 18.75,
    },
    148: {
        "name": "FLASH",
        "effect": "ACCURACY_DOWN1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 70,
        "pp": 20,
        "power": 9.25,
    },
    149: {
        "name": "PSYWAVE",
        "effect": "SPECIAL_DAMAGE_EFFECT",
        "raw_power": 1,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 80,
        "pp": 15,
        "power": 36.515625,
    },
    150: {
        "name": "SPLASH",
        "effect": "SPLASH_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 40,
        "power": 0.0,
    },
    151: {
        "name": "ACID_ARMOR",
        "effect": "DEFENSE_UP2_EFFECT",
        "raw_power": 0,
        "type": "POISON",
        "type_id": 3,
        "accuracy": 100,
        "pp": 40,
        "power": 20.0,
    },
    152: {
        "name": "CRABHAMMER",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 90,
        "type": "WATER",
        "type_id": 21,
        "accuracy": 85,
        "pp": 10,
        "power": 75.796875,
    },
    153: {
        "name": "EXPLOSION",
        "effect": "EXPLODE_EFFECT",
        "raw_power": 170,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 5,
        "power": 45.5,
    },
    154: {
        "name": "FURY_SWIPES",
        "effect": "TWO_TO_FIVE_ATTACKS_EFFECT",
        "raw_power": 18,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 80,
        "pp": 15,
        "power": 48.09375,
    },
    155: {
        "name": "BONEMERANG",
        "effect": "ATTACK_TWICE_EFFECT",
        "raw_power": 50,
        "type": "GROUND",
        "type_id": 4,
        "accuracy": 90,
        "pp": 10,
        "power": 85.3125,
    },
    156: {
        "name": "REST",
        "effect": "HEAL_EFFECT",
        "raw_power": 0,
        "type": "PSYCHIC_TYPE",
        "type_id": 24,
        "accuracy": 100,
        "pp": 10,
        "power": 17.5,
    },
    157: {
        "name": "ROCK_SLIDE",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 75,
        "type": "ROCK",
        "type_id": 5,
        "accuracy": 90,
        "pp": 10,
        "power": 63.984375,
    },
    158: {
        "name": "HYPER_FANG",
        "effect": "FLINCH_SIDE_EFFECT1",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 90,
        "pp": 15,
        "power": 91.40625,
    },
    159: {
        "name": "SHARPEN",
        "effect": "ATTACK_UP1_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    160: {
        "name": "CONVERSION",
        "effect": "CONVERSION_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 30,
        "power": 10.0,
    },
    161: {
        "name": "TRI_ATTACK",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 80,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 70.0,
    },
    162: {
        "name": "SUPER_FANG",
        "effect": "SUPER_FANG_EFFECT",
        "raw_power": 1,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 90,
        "pp": 10,
        "power": 34.978125,
    },
    163: {
        "name": "SLASH",
        "effect": "NO_ADDITIONAL_EFFECT",
        "raw_power": 70,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 20,
        "power": 70.0,
    },
    164: {
        "name": "SUBSTITUTE",
        "effect": "SUBSTITUTE_EFFECT",
        "raw_power": 0,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 35.0,
    },
    165: {
        "name": "STRUGGLE",
        "effect": "RECOIL_EFFECT",
        "raw_power": 50,
        "type": "NORMAL",
        "type_id": 0,
        "accuracy": 100,
        "pp": 10,
        "power": 39.375,
    },
}

ITEM_PRICES = {
    "MASTER_BALL": 0,
    "ULTRA_BALL": 1200,
    "GREAT_BALL": 600,
    "POKE_BALL": 200,
    "TOWN_MAP": 0,
    "BICYCLE": 0,
    "SURFBOARD": 0,
    "SAFARI_BALL": 1000,
    "POKEDEX": 0,
    "MOON_STONE": 0,
    "ANTIDOTE": 100,
    "BURN_HEAL": 250,
    "ICE_HEAL": 250,
    "AWAKENING": 200,
    "PARLYZ_HEAL": 200,
    "FULL_RESTORE": 3000,
    "MAX_POTION": 2500,
    "HYPER_POTION": 1500,
    "SUPER_POTION": 700,
    "POTION": 300,
    "BOULDERBADGE": 0,
    "CASCADEBADGE": 0,
    "THUNDERBADGE": 0,
    "RAINBOWBADGE": 0,
    "SOULBADGE": 0,
    "MARSHBADGE": 0,
    "VOLCANOBADGE": 0,
    "EARTHBADGE": 0,
    "ESCAPE_ROPE": 550,
    "REPEL": 350,
    "OLD_AMBER": 0,
    "FIRE_STONE": 2100,
    "THUNDER_STONE": 2100,
    "WATER_STONE": 2100,
    "HP_UP": 9800,
    "PROTEIN": 9800,
    "IRON": 9800,
    "CARBOS": 9800,
    "CALCIUM": 9800,
    "RARE_CANDY": 4800,
    "DOME_FOSSIL": 0,
    "HELIX_FOSSIL": 0,
    "SECRET_KEY": 0,
    "XXX": 9800,
    "BIKE_VOUCHER": 0,
    "X_ACCURACY": 950,
    "LEAF_STONE": 2100,
    "CARD_KEY": 0,
    "NUGGET": 10000,
    "POKE_DOLL": 1000,
    "FULL_HEAL": 600,
    "REVIVE": 1500,
    "MAX_REVIVE": 4000,
    "GUARD_SPEC": 700,
    "SUPER_REPEL": 500,
    "MAX_REPEL": 700,
    "DIRE_HIT": 650,
    "COIN": 10,
    "FRESH_WATER": 200,
    "SODA_POP": 300,
    "LEMONADE": 350,
    "S_S_TICKET": 0,
    "GOLD_TEETH": 0,
    "X_ATTACK": 500,
    "X_DEFEND": 550,
    "X_SPEED": 350,
    "X_SPECIAL": 350,
    "COIN_CASE": 0,
    "OAKS_PARCEL": 0,
    "ITEMFINDER": 0,
    "SILPH_SCOPE": 0,
    "POKE_FLUTE": 0,
    "LIFT_KEY": 0,
    "EXP_ALL": 0,
    "OLD_ROD": 0,
    "GOOD_ROD": 0,
    "SUPER_ROD": 0,
    "PP_UP": 0,
    "ETHER": 0,
    "MAX_ETHER": 0,
    "ELIXER": 0,
    "MAX_ELIXER": 0,
    "FLOOR_B2F": 0,
    "FLOOR_B1F": 0,
    "FLOOR_1F": 0,
    "FLOOR_2F": 0,
    "FLOOR_3F": 0,
    "FLOOR_4F": 0,
    "FLOOR_5F": 0,
    "FLOOR_6F": 0,
    "FLOOR_7F": 0,
    "FLOOR_8F": 0,
    "FLOOR_9F": 0,
    "FLOOR_10F": 0,
    "FLOOR_11F": 0,
    "FLOOR_B4F": 0,
}

TM_PRICES = {
    "TM01": 3000,
    "TM02": 2000,
    "TM03": 2000,
    "TM04": 1000,
    "TM05": 3000,
    "TM06": 4000,
    "TM07": 2000,
    "TM08": 4000,
    "TM09": 3000,
    "TM10": 4000,
    "TM11": 2000,
    "TM12": 1000,
    "TM13": 4000,
    "TM14": 5000,
    "TM15": 5000,
    "TM16": 5000,
    "TM17": 3000,
    "TM18": 2000,
    "TM19": 3000,
    "TM20": 2000,
    "TM21": 5000,
    "TM22": 5000,
    "TM23": 5000,
    "TM24": 2000,
    "TM25": 5000,
    "TM26": 4000,
    "TM27": 5000,
    "TM28": 2000,
    "TM29": 4000,
    "TM30": 1000,
    "TM31": 2000,
    "TM32": 1000,
    "TM33": 1000,
    "TM34": 2000,
    "TM35": 4000,
    "TM36": 2000,
    "TM37": 2000,
    "TM38": 5000,
    "TM39": 2000,
    "TM40": 4000,
    "TM41": 2000,
    "TM42": 2000,
    "TM43": 5000,
    "TM44": 2000,
    "TM45": 2000,
    "TM46": 4000,
    "TM47": 3000,
    "TM48": 4000,
    "TM49": 4000,
    "TM50": 2000,
}

# ITEM_TM_IDS_PRICES = {1: 0, 2: 1200, 3: 600, 4: 200, 5: 0, 6: 0, 7: 0, 8: 1000, 9: 0, 10: 0, 11: 100, 12: 250, 13: 250, 14: 200, 15: 200, 16: 3000, 17: 2500, 18: 1500, 19: 700, 20: 300, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 550, 30: 350, 31: 0, 32: 2100, 33: 2100, 34: 2100, 35: 9800, 36: 9800, 37: 9800, 38: 9800, 39: 9800, 40: 4800, 41: 0, 42: 0, 43: 0, 45: 0, 46: 950, 47: 2100, 48: 0, 49: 10000, 51: 1000, 52: 600, 53: 1500, 54: 4000, 55: 700, 56: 500, 57: 700, 58: 650, 59: 10, 60: 200, 61: 300, 62: 350, 63: 0, 64: 0, 65: 500, 66: 550, 67: 350, 68: 350, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 201: 3000, 202: 2000, 203: 2000, 204: 1000, 205: 3000, 206: 4000, 207: 2000, 208: 4000, 209: 3000, 210: 4000, 211: 2000, 212: 1000, 213: 4000, 214: 5000, 215: 5000, 216: 5000, 217: 3000, 218: 2000, 219: 3000, 220: 2000, 221: 5000, 222: 5000, 223: 5000, 224: 2000, 225: 5000, 226: 4000, 227: 5000, 228: 2000, 229: 4000, 230: 1000, 231: 2000, 232: 1000, 233: 1000, 234: 2000, 235: 4000, 236: 2000, 237: 2000, 238: 5000, 239: 2000, 240: 4000, 241: 2000, 242: 2000, 243: 5000, 244: 2000, 245: 2000, 246: 4000, 247: 3000, 248: 4000, 249: 4000, 250: 2000}
ITEM_TM_IDS_PRICES = {
    2: 1200,
    3: 600,
    4: 200,
    8: 1000,
    10: 0,
    11: 100,
    12: 250,
    13: 250,
    14: 200,
    15: 200,
    16: 3000,
    17: 2500,
    18: 1500,
    19: 700,
    20: 300,
    29: 550,
    30: 350,
    32: 2100,
    33: 2100,
    34: 2100,
    35: 9800,
    36: 9800,
    37: 9800,
    38: 9800,
    39: 9800,
    40: 4800,
    46: 950,
    47: 2100,
    49: 10000,
    51: 1000,
    52: 600,
    53: 1500,
    54: 4000,
    55: 700,
    56: 500,
    57: 700,
    58: 650,
    59: 10,
    60: 200,
    61: 300,
    62: 350,
    65: 500,
    66: 550,
    67: 350,
    68: 350,
    76: 0,
    77: 0,
    78: 0,
    79: 0,
    80: 0,
    81: 0,
    201: 3000,
    202: 2000,
    203: 2000,
    204: 1000,
    205: 3000,
    206: 4000,
    207: 2000,
    208: 4000,
    209: 3000,
    210: 4000,
    211: 2000,
    212: 1000,
    213: 4000,
    214: 5000,
    215: 5000,
    216: 5000,
    217: 3000,
    218: 2000,
    219: 3000,
    220: 2000,
    221: 5000,
    222: 5000,
    223: 5000,
    224: 2000,
    225: 5000,
    226: 4000,
    227: 5000,
    228: 2000,
    229: 4000,
    230: 1000,
    231: 2000,
    232: 1000,
    233: 1000,
    234: 2000,
    235: 4000,
    236: 2000,
    237: 2000,
    238: 5000,
    239: 2000,
    240: 4000,
    241: 2000,
    242: 2000,
    243: 5000,
    244: 2000,
    245: 2000,
    246: 4000,
    247: 3000,
    248: 4000,
    249: 4000,
    250: 2000,
}

MART_ITEMS_NAME_DICT = {
    "42@2,5": {
        "items": ["POKE_BALL", "ANTIDOTE", "PARLYZ_HEAL", "BURN_HEAL"],
        "map": "VIRIDIAN_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "56@2,5": {
        "items": [
            "POKE_BALL",
            "POTION",
            "ESCAPE_ROPE",
            "ANTIDOTE",
            "BURN_HEAL",
            "AWAKENING",
            "PARLYZ_HEAL",
        ],
        "map": "PEWTER_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "67@2,5": {
        "items": [
            "POKE_BALL",
            "POTION",
            "REPEL",
            "ANTIDOTE",
            "BURN_HEAL",
            "AWAKENING",
            "PARLYZ_HEAL",
        ],
        "map": "CERULEAN_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "91@2,5": {
        "items": ["POKE_BALL", "SUPER_POTION", "ICE_HEAL", "AWAKENING", "PARLYZ_HEAL", "REPEL"],
        "map": "VERMILION_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "150@2,5": {
        "items": [
            "GREAT_BALL",
            "SUPER_POTION",
            "REVIVE",
            "ESCAPE_ROPE",
            "SUPER_REPEL",
            "ANTIDOTE",
            "BURN_HEAL",
            "ICE_HEAL",
            "PARLYZ_HEAL",
        ],
        "map": "LAVENDER_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "123@5,5": {
        "items": [
            "GREAT_BALL",
            "SUPER_POTION",
            "REVIVE",
            "SUPER_REPEL",
            "ANTIDOTE",
            "BURN_HEAL",
            "ICE_HEAL",
            "AWAKENING",
            "PARLYZ_HEAL",
        ],
        "map": "CELADON_MART_2F",
        "x": 5,
        "y": 5,
        "dir": "up",
    },
    "123@6,5": {
        "items": [
            "TM_DOUBLE_TEAM",
            "TM_REFLECT",
            "TM_RAZOR_WIND",
            "TM_HORN_DRILL",
            "TM_EGG_BOMB",
            "TM_MEGA_PUNCH",
            "TM_MEGA_KICK",
            "TM_TAKE_DOWN",
            "TM_SUBMISSION",
        ],
        "map": "CELADON_MART_2F",
        "x": 6,
        "y": 5,
        "dir": "up",
    },
    "125@5,5": {
        "items": ["POKE_DOLL", "FIRE_STONE", "THUNDER_STONE", "WATER_STONE", "LEAF_STONE"],
        "map": "CELADON_MART_4F",
        "x": 5,
        "y": 5,
        "dir": "down",
    },
    "136@5,5": {
        "items": [
            "X_ACCURACY",
            "GUARD_SPEC",
            "DIRE_HIT",
            "X_ATTACK",
            "X_DEFEND",
            "X_SPEED",
            "X_SPECIAL",
        ],
        "map": "CELADON_MART_5F",
        "x": 5,
        "y": 5,
        "dir": "up",
    },
    "136@6,5": {
        "items": ["HP_UP", "PROTEIN", "IRON", "CARBOS", "CALCIUM"],
        "map": "CELADON_MART_5F",
        "x": 6,
        "y": 5,
        "dir": "up",
    },
    "152@2,5": {
        "items": ["ULTRA_BALL", "GREAT_BALL", "SUPER_POTION", "REVIVE", "FULL_HEAL", "SUPER_REPEL"],
        "map": "FUCHSIA_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "172@2,5": {
        "items": [
            "ULTRA_BALL",
            "GREAT_BALL",
            "HYPER_POTION",
            "MAX_REPEL",
            "ESCAPE_ROPE",
            "FULL_HEAL",
            "REVIVE",
        ],
        "map": "CINNABAR_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "180@2,5": {
        "items": ["GREAT_BALL", "HYPER_POTION", "MAX_REPEL", "ESCAPE_ROPE", "FULL_HEAL", "REVIVE"],
        "map": "SAFFRON_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "174@2,5": {
        "items": [
            "ULTRA_BALL",
            "GREAT_BALL",
            "FULL_RESTORE",
            "MAX_POTION",
            "FULL_HEAL",
            "REVIVE",
            "MAX_REPEL",
        ],
        "map": "INDIGO_PLATEAU_LOBBY",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
}

MART_ITEMS_ID_DICT = {
    "42@2,5": {"items": [4, 11, 15, 12], "map": "VIRIDIAN_MART", "x": 2, "y": 5, "dir": "left"},
    "56@2,5": {
        "items": [4, 20, 29, 11, 12, 14, 15],
        "map": "PEWTER_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "67@2,5": {
        "items": [4, 20, 30, 11, 12, 14, 15],
        "map": "CERULEAN_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "91@2,5": {
        "items": [4, 19, 13, 14, 15, 30],
        "map": "VERMILION_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "150@2,5": {
        "items": [3, 19, 53, 29, 56, 11, 12, 13, 15],
        "map": "LAVENDER_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "123@5,5": {
        "items": [3, 19, 53, 56, 11, 12, 13, 14, 15],
        "map": "CELADON_MART_2F",
        "x": 5,
        "y": 5,
        "dir": "up",
    },
    "123@6,5": {"items": [], "map": "CELADON_MART_2F", "x": 6, "y": 5, "dir": "up"},
    "125@5,5": {
        "items": [51, 32, 33, 34, 47],
        "map": "CELADON_MART_4F",
        "x": 5,
        "y": 5,
        "dir": "down",
    },
    "136@5,5": {
        "items": [46, 55, 58, 65, 66, 67, 68],
        "map": "CELADON_MART_5F",
        "x": 5,
        "y": 5,
        "dir": "up",
    },
    "136@6,5": {
        "items": [35, 36, 37, 38, 39],
        "map": "CELADON_MART_5F",
        "x": 6,
        "y": 5,
        "dir": "up",
    },
    "152@2,5": {
        "items": [2, 3, 19, 53, 52, 56],
        "map": "FUCHSIA_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "172@2,5": {
        "items": [2, 3, 18, 57, 29, 52, 53],
        "map": "CINNABAR_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "180@2,5": {
        "items": [3, 18, 57, 29, 52, 53],
        "map": "SAFFRON_MART",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
    "174@2,5": {
        "items": [2, 3, 16, 17, 52, 53, 57],
        "map": "INDIGO_PLATEAU_LOBBY",
        "x": 2,
        "y": 5,
        "dir": "left",
    },
}

ITEM_NAME_TO_ID_DICT = {
    "NO_ITEM": 0,
    "MASTER_BALL": 1,
    "ULTRA_BALL": 2,
    "GREAT_BALL": 3,
    "POKE_BALL": 4,
    "TOWN_MAP": 5,
    "BICYCLE": 6,
    "SURFBOARD": 7,
    "SAFARI_BALL": 8,
    "POKEDEX": 9,
    "MOON_STONE": 10,
    "ANTIDOTE": 11,
    "BURN_HEAL": 12,
    "ICE_HEAL": 13,
    "AWAKENING": 14,
    "PARLYZ_HEAL": 15,
    "FULL_RESTORE": 16,
    "MAX_POTION": 17,
    "HYPER_POTION": 18,
    "SUPER_POTION": 19,
    "POTION": 20,
    "BOULDERBADGE": 21,
    "CASCADEBADGE": 22,
    "THUNDERBADGE": 23,
    "RAINBOWBADGE": 24,
    "SOULBADGE": 25,
    "MARSHBADGE": 26,
    "VOLCANOBADGE": 27,
    "EARTHBADGE": 28,
    "ESCAPE_ROPE": 29,
    "REPEL": 30,
    "OLD_AMBER": 31,
    "FIRE_STONE": 32,
    "THUNDER_STONE": 33,
    "WATER_STONE": 34,
    "HP_UP": 35,
    "PROTEIN": 36,
    "IRON": 37,
    "CARBOS": 38,
    "CALCIUM": 39,
    "RARE_CANDY": 40,
    "DOME_FOSSIL": 41,
    "HELIX_FOSSIL": 42,
    "SECRET_KEY": 43,
    "UNUSED_ITEM": 44,
    "BIKE_VOUCHER": 45,
    "X_ACCURACY": 46,
    "LEAF_STONE": 47,
    "CARD_KEY": 48,
    "NUGGET": 49,
    "PP_UP_2": 50,
    "POKE_DOLL": 51,
    "FULL_HEAL": 52,
    "REVIVE": 53,
    "MAX_REVIVE": 54,
    "GUARD_SPEC": 55,
    "SUPER_REPEL": 56,
    "MAX_REPEL": 57,
    "DIRE_HIT": 58,
    "COIN": 59,
    "FRESH_WATER": 60,
    "SODA_POP": 61,
    "LEMONADE": 62,
    "S_S_TICKET": 63,
    "GOLD_TEETH": 64,
    "X_ATTACK": 65,
    "X_DEFEND": 66,
    "X_SPEED": 67,
    "X_SPECIAL": 68,
    "COIN_CASE": 69,
    "OAKS_PARCEL": 70,
    "ITEMFINDER": 71,
    "SILPH_SCOPE": 72,
    "POKE_FLUTE": 73,
    "LIFT_KEY": 74,
    "EXP_ALL": 75,
    "OLD_ROD": 76,
    "GOOD_ROD": 77,
    "SUPER_ROD": 78,
    "PP_UP": 79,
    "ETHER": 80,
    "MAX_ETHER": 81,
    "ELIXER": 82,
    "MAX_ELIXER": 83,
    "FLOOR_B2F": 84,
    "FLOOR_B1F": 85,
    "FLOOR_1F": 86,
    "FLOOR_2F": 87,
    "FLOOR_3F": 88,
    "FLOOR_4F": 89,
    "FLOOR_5F": 90,
    "FLOOR_6F": 91,
    "FLOOR_7F": 92,
    "FLOOR_8F": 93,
    "FLOOR_9F": 94,
    "FLOOR_10F": 95,
    "FLOOR_11F": 96,
    "FLOOR_B4F": 97,
}

MART_MAP_IDS = [42, 56, 67, 91, 150, 123, 125, 136, 152, 172, 180, 174]


ALL_HM_IDS_RAM = [
    0xC4,  # CUT
    0xC5,  # FLY
    0xC6,  # SURF
    0xC7,  # STRENGTH
    0xC8,  # FLASH
]

ALL_KEY_ITEMS_RAM = [
    0x05,  # TOWN_MAP
    0x06,  # BICYCLE
    0x07,  # SURFBOARD
    0x08,  # SAFARI_BALL
    0x09,  # POKEDEX
    0x15,  # BOULDERBADGE
    0x16,  # CASCADEBADGE
    0x17,  # THUNDERBADGE
    0x18,  # RAINBOWBADGE
    0x19,  # SOULBADGE
    0x1A,  # MARSHBADGE
    0x1B,  # VOLCANOBADGE
    0x1C,  # EARTHBADGE
    0x20,  # OLD_AMBER
    0x29,  # DOME_FOSSIL
    0x2A,  # HELIX_FOSSIL
    0x2B,  # SECRET_KEY
    0x2D,  # BIKE_VOUCHER
    0x30,  # CARD_KEY
    0x3F,  # S_S_TICKET
    0x40,  # GOLD_TEETH
    0x45,  # COIN_CASE
    0x46,  # OAKS_PARCEL
    0x47,  # ITEMFINDER
    0x48,  # SILPH_SCOPE
    0x49,  # POKE_FLUTE
    0x4A,  # LIFT_KEY
    0x4C,  # OLD_ROD
    0x4D,  # GOOD_ROD
    0x4E,  # SUPER_ROD
    # quest items to keep
    0x3C,  # FRESH_WATER
    0x3D,  # SODA_POP
    0x3E,  # LEMONADE
]

SPECIAL_KEY_ITEM_IDS_RAM = [
    0x30,  # CARD_KEY
    0x2B,  # SECRET_KEY
    0x48,  # SILPH_SCOPE
    0x4A,  # LIFT_KEY
    0x49,  # POKE_FLUTE
    0x3F,  # S_S_TICKET
    0x06,  # BICYCLE
    0x40,  # GOLD_TEETH
    0x3C,  # FRESH_WATER
    0x3D,  # SODA_POP
    0x3E,  # LEMONADE
]

ALL_GOOD_ITEMS_RAM = [ALL_HM_IDS_RAM, ALL_KEY_ITEMS_RAM, SPECIAL_KEY_ITEM_IDS_RAM]

ALL_HM_IDS_STR = [
    "HM01 Cut",  # CUT
    "HM02 Fly",  # FLY
    "HM03 Surf",  # SURF
    "HM04 Strength",  # STRENGTH
    "HM05 Flash",  # FLASH
]

ALL_KEY_ITEMS_STR = [
    "Town Map",  # TOWN_MAP
    "Bicycle",  # BICYCLE
    "Surfboard",  # SURFBOARD
    "Safari Ball",  # SAFARI_BALL
    "PokeDex",  # POKEDEX
    "BoulderBadge",  # BOULDERBADGE
    "CascadeBadge",  # CASCADEBADGE
    "ThunderBadge",  # THUNDERBADGE
    "RainbowBadge",  # RAINBOWBADGE
    "SoulBadge",  # SOULBADGE
    "MarshBadge",  # MARSHBADGE
    "VolcanoBadge",  # VOLCANOBADGE
    "EarthBadge",  # EARTHBADGE
    "Old Amber",  # OLD_AMBER
    "Dome Fossil",  # DOME_FOSSIL
    "Helix Fossil",  # HELIX_FOSSIL
    "Secret Key",  # SECRET_KEY
    "Bike Voucher",  # BIKE_VOUCHER
    "Card Key",  # CARD_KEY
    "S.S. Ticket",  # S_S_TICKET
    "Gold Teeth",  # GOLD_TEETH
    "Coin Case",  # COIN_CASE
    "Oak's Parcel",  # OAKS_PARCEL
    "Item Finder",  # ITEMFINDER
    "Silph Scope",  # SILPH_SCOPE
    "Poke Flute",  # POKE_FLUTE
    "Lift Key",  # LIFT_KEY
    "Old Rod",  # OLD_ROD
    "Good Rod",  # GOOD_ROD
    "Super Rod",  # SUPER_ROD
    # quest items to keep
    "Fresh Water",  # FRESH_WATER
    "Soda Pop",  # SODA_POP
    "Lemonade",  # LEMONADE
]

SPECIAL_KEY_ITEM_IDS_STR = [
    "Card Key",  # CARD_KEY
    "Secret Key",  # SECRET_KEY
    "Silph Scope",  # SILPH_SCOPE
    "Lift Key",  # LIFT_KEY
    "Poke Flute",  # POKE_FLUTE
    "S.S. Ticket",  # S_S_TICKET
    "Bicycle",  # BICYCLE
    "Gold Teeth",  # GOLD_TEETH
    "Fresh Water",  # FRESH_WATER
    "Soda Pop",  # SODA_POP
    "Lemonade",  # LEMONADE
]

ALL_GOOD_ITEMS_STR = [ALL_HM_IDS_STR, ALL_KEY_ITEMS_STR, SPECIAL_KEY_ITEM_IDS_STR]


PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)

VISITED_MASK_SHAPE = (144 // 16, 160 // 16, 1)

TM_HM_MOVES = set(
    [
        5,  # Mega punch
        0xD,  # Razor wind
        0xE,  # Swords dance
        0x12,  # Whirlwind
        0x19,  # Mega kick
        0x5C,  # Toxic
        0x20,  # Horn drill
        0x22,  # Body slam
        0x24,  # Take down
        0x26,  # Double edge
        0x3D,  # Bubble beam
        0x37,  # Water gun
        0x3A,  # Ice beam
        0x3B,  # Blizzard
        0x3F,  # Hyper beam
        0x06,  # Pay day
        0x42,  # Submission
        0x44,  # Counter
        0x45,  # Seismic toss
        0x63,  # Rage
        0x48,  # Mega drain
        0x4C,  # Solar beam
        0x52,  # Dragon rage
        0x55,  # Thunderbolt
        0x57,  # Thunder
        0x59,  # Earthquake
        0x5A,  # Fissure
        0x5B,  # Dig
        0x5E,  # Psychic
        0x64,  # Teleport
        0x66,  # Mimic
        0x68,  # Double team
        0x73,  # Reflect
        0x75,  # Bide
        0x76,  # Metronome
        0x78,  # Selfdestruct
        0x79,  # Egg bomb
        0x7E,  # Fire blast
        0x81,  # Swift
        0x82,  # Skull bash
        0x87,  # Softboiled
        0x8A,  # Dream eater
        0x8F,  # Sky attack
        0x9C,  # Rest
        0x56,  # Thunder wave
        0x95,  # Psywave
        0x99,  # Explosion
        0x9D,  # Rock slide
        0xA1,  # Tri attack
        0xA4,  # Substitute
        0x0F,  # Cut
        0x13,  # Fly
        0x39,  # Surf
        0x46,  # Strength
        0x94,  # Flash
    ]
)

HM_ITEM_IDS = set([0xC4, 0xC5, 0xC6, 0xC7, 0xC8])

RESET_MAP_IDS = set(
    [
        0x0,  # Pallet Town
        0x1,  # Viridian City
        0x2,  # Pewter City
        0x3,  # Cerulean City
        0x4,  # Lavender Town
        0x5,  # Vermilion City
        0x6,  # Celadon City
        0x7,  # Fuchsia City
        0x8,  # Cinnabar Island
        0x9,  # Indigo Plateau
        0xA,  # Saffron City
        0xF,  # Route 4 (Mt Moon)
        0x10,  # Route 10 (Rock Tunnel)
        0xE9,  # Silph Co 9F (Heal station)
    ]
)

CUT_SPECIES_IDS = {
    0x99,
    0x09,
    0x9A,
    0xB0,
    0xB2,
    0xB4,
    0x72,
    0x60,
    0x61,
    0xB9,
    0xBA,
    0xBB,
    0x6D,
    0x2E,
    0xBC,
    0xBD,
    0xBE,
    0x18,
    0x9B,
    0x40,
    0x4E,
    0x8A,
    0x0B,
    0x1E,
    0x1A,
    0x1D,
    0x5B,
    0x15,
}

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))

FIELD_MOVES = ["CUT", "FLY", "SURF", "SURF", "STRENGTH", "FLASH", "DIG", "TELEPORT", "SOFTBOILED"]

FIELD_MOVES_MAP = {i + 1: v for i, v in enumerate(FIELD_MOVES)}

MAX_ITEM_CAPACITY = 20

# Starts at 0x1
KEY_ITEM_IDS = [
    False,  # MASTER_BALL
    False,  # ULTRA_BALL
    False,  # GREAT_BALL
    False,  # POKE_BALL
    True,  # TOWN_MAP
    True,  # BICYCLE
    True,  # SURFBOARD
    True,  # SAFARI_BALL
    True,  # POKEDEX
    False,  # MOON_STONE
    False,  # ANTIDOTE
    False,  # BURN_HEAL
    False,  # ICE_HEAL
    False,  # AWAKENING
    False,  # PARLYZ_HEAL
    False,  # FULL_RESTORE
    False,  # MAX_POTION
    False,  # HYPER_POTION
    False,  # SUPER_POTION
    False,  # POTION
    True,  # BOULDERBADGE
    True,  # CASCADEBADGE
    True,  # THUNDERBADGE
    True,  # RAINBOWBADGE
    True,  # SOULBADGE
    True,  # MARSHBADGE
    True,  # VOLCANOBADGE
    True,  # EARTHBADGE
    False,  # ESCAPE_ROPE
    False,  # REPEL
    True,  # OLD_AMBER
    False,  # FIRE_STONE
    False,  # THUNDER_STONE
    False,  # WATER_STONE
    False,  # HP_UP
    False,  # PROTEIN
    False,  # IRON
    False,  # CARBOS
    False,  # CALCIUM
    False,  # RARE_CANDY
    True,  # DOME_FOSSIL
    True,  # HELIX_FOSSIL
    True,  # SECRET_KEY
    True,  # ITEM_2C
    True,  # BIKE_VOUCHER
    False,  # X_ACCURACY
    False,  # LEAF_STONE
    True,  # CARD_KEY
    False,  # NUGGET
    False,  # ITEM_32
    False,  # POKE_DOLL
    False,  # FULL_HEAL
    False,  # REVIVE
    False,  # MAX_REVIVE
    False,  # GUARD_SPEC
    False,  # SUPER_REPEL
    False,  # MAX_REPEL
    False,  # DIRE_HIT
    False,  # COIN
    False,  # FRESH_WATER
    False,  # SODA_POP
    False,  # LEMONADE
    True,  # S_S_TICKET
    True,  # GOLD_TEETH
    False,  # X_ATTACK
    False,  # X_DEFEND
    False,  # X_SPEED
    False,  # X_SPECIAL
    True,  # COIN_CASE
    True,  # OAKS_PARCEL
    True,  # ITEMFINDER
    True,  # SILPH_SCOPE
    True,  # POKE_FLUTE
    True,  # LIFT_KEY
    False,  # EXP_ALL
    True,  # OLD_ROD
    True,  # GOOD_ROD
    True,  # SUPER_ROD
    False,  # PP_UP
    False,  # ETHER
    False,  # MAX_ETHER
    False,  # ELIXER
    False,  # MAX_ELIXER
]

# Start = 0x1
ITEM_NAMES = [
    "MASTER_BALL",
    "ULTRA_BALL",
    "GREAT_BALL",
    "POKE_BALL",
    "TOWN_MAP",
    "BICYCLE",
    "SURFBOARD",
    "SAFARI_BALL",
    "POKEDEX",
    "MOON_STONE",
    "ANTIDOTE",
    "BURN_HEAL",
    "ICE_HEAL",
    "AWAKENING",
    "PARLYZ_HEAL",
    "FULL_RESTORE",
    "MAX_POTION",
    "HYPER_POTION",
    "SUPER_POTION",
    "POTION",
    "BOULDERBADGE",
    "CASCADEBADGE",
    "THUNDERBADGE",
    "RAINBOWBADGE",
    "SOULBADGE",
    "MARSHBADGE",
    "VOLCANOBADGE",
    "EARTHBADGE",
    "ESCAPE_ROPE",
    "REPEL",
    "OLD_AMBER",
    "FIRE_STONE",
    "THUNDER_STONE",
    "WATER_STONE",
    "HP_UP",
    "PROTEIN",
    "IRON",
    "CARBOS",
    "CALCIUM",
    "RARE_CANDY",
    "DOME_FOSSIL",
    "HELIX_FOSSIL",
    "SECRET_KEY",
    "UNUSED_ITEM",
    "BIKE_VOUCHER",
    "X_ACCURACY",
    "LEAF_STONE",
    "CARD_KEY",
    "NUGGET",
    "PP_UP_2",
    "POKE_DOLL",
    "FULL_HEAL",
    "REVIVE",
    "MAX_REVIVE",
    "GUARD_SPEC",
    "SUPER_REPEL",
    "MAX_REPEL",
    "DIRE_HIT",
    "COIN",
    "FRESH_WATER",
    "SODA_POP",
    "LEMONADE",
    "S_S_TICKET",
    "GOLD_TEETH",
    "X_ATTACK",
    "X_DEFEND",
    "X_SPEED",
    "X_SPECIAL",
    "COIN_CASE",
    "OAKS_PARCEL",
    "ITEMFINDER",
    "SILPH_SCOPE",
    "POKE_FLUTE",
    "LIFT_KEY",
    "EXP_ALL",
    "OLD_ROD",
    "GOOD_ROD",
    "SUPER_ROD",
    "PP_UP",
    "ETHER",
    "MAX_ETHER",
    "ELIXER",
    "MAX_ELIXER",
    "FLOOR_B2F",
    "FLOOR_B1F",
    "FLOOR_1F",
    "FLOOR_2F",
    "FLOOR_3F",
    "FLOOR_4F",
    "FLOOR_5F",
    "FLOOR_6F",
    "FLOOR_7F",
    "FLOOR_8F",
    "FLOOR_9F",
    "FLOOR_10F",
    "FLOOR_11F",
    "FLOOR_B4F",
]

# Start = 0xC4
TM_HM_ITEM_IDS = [
    "HM_01",
    "HM_02",
    "HM_03",
    "HM_04",
    "HM_05",
    "TM_01",
    "TM_02",
    "TM_03",
    "TM_04",
    "TM_05",
    "TM_06",
    "TM_07",
    "TM_08",
    "TM_09",
    "TM_10",
    "TM_11",
    "TM_12",
    "TM_13",
    "TM_14",
    "TM_15",
    "TM_16",
    "TM_17",
    "TM_18",
    "TM_19",
    "TM_20",
    "TM_21",
    "TM_22",
    "TM_23",
    "TM_24",
    "TM_25",
    "TM_26",
    "TM_27",
    "TM_28",
    "TM_29",
    "TM_30",
    "TM_31",
    "TM_32",
    "TM_33",
    "TM_34",
    "TM_35",
    "TM_36",
    "TM_37",
    "TM_38",
    "TM_39",
    "TM_40",
    "TM_41",
    "TM_42",
    "TM_43",
    "TM_44",
    "TM_45",
    "TM_46",
    "TM_47",
    "TM_48",
    "TM_49",
    "TM_50",
]

HM_ITEM_IDS = {0xC4, 0xC5, 0xC6, 0xC7, 0xC8}

ITEM_NAME_TO_ID = (
    {v: i + 0x1 for i, v in enumerate(ITEM_NAMES)}
    | {v: i + 0xC4 for i, v in enumerate(TM_HM_ITEM_IDS)}
    | {"SAFARI_BAIT": 0x15, "SAFARI_ROCK": 0x16}
)

RESET_MAP_IDS = {
    0x0,  # Pallet Town
    0x1,  # Viridian City
    0x2,  # Pewter City
    0x3,  # Cerulean City
    0x4,  # Lavender Town
    0x5,  # Vermilion City
    0x6,  # Celadon City
    0x7,  # Fuchsia City
    0x8,  # Cinnabar Island
    0x9,  # Indigo Plateau
    0xA,  # Saffron City
    0xF,  # Route 4 (Mt Moon)
    0x10,  # Route 10 (Rock Tunnel)
    0xE9,  # Silph Co 9F (Heal station)
}

SPECIES_IDS = {
    "RHYDON": 0x01,
    "KANGASKHAN": 0x02,
    "NIDORAN_M": 0x03,
    "CLEFAIRY": 0x04,
    "SPEAROW": 0x05,
    "VOLTORB": 0x06,
    "NIDOKING": 0x07,
    "SLOWBRO": 0x08,
    "IVYSAUR": 0x09,
    "EXEGGUTOR": 0x0A,
    "LICKITUNG": 0x0B,
    "EXEGGCUTE": 0x0C,
    "GRIMER": 0x0D,
    "GENGAR": 0x0E,
    "NIDORAN_F": 0x0F,
    "NIDOQUEEN": 0x10,
    "CUBONE": 0x11,
    "RHYHORN": 0x12,
    "LAPRAS": 0x13,
    "ARCANINE": 0x14,
    "MEW": 0x15,
    "GYARADOS": 0x16,
    "SHELLDER": 0x17,
    "TENTACOOL": 0x18,
    "GASTLY": 0x19,
    "SCYTHER": 0x1A,
    "STARYU": 0x1B,
    "BLASTOISE": 0x1C,
    "PINSIR": 0x1D,
    "TANGELA": 0x1E,
    "MISSINGNO_1F": 0x1F,
    "MISSINGNO_20": 0x20,
    "GROWLITHE": 0x21,
    "ONIX": 0x22,
    "FEAROW": 0x23,
    "PIDGEY": 0x24,
    "SLOWPOKE": 0x25,
    "KADABRA": 0x26,
    "GRAVELER": 0x27,
    "CHANSEY": 0x28,
    "MACHOKE": 0x29,
    "MR_MIME": 0x2A,
    "HITMONLEE": 0x2B,
    "HITMONCHAN": 0x2C,
    "ARBOK": 0x2D,
    "PARASECT": 0x2E,
    "PSYDUCK": 0x2F,
    "DROWZEE": 0x30,
    "GOLEM": 0x31,
    "MISSINGNO_32": 0x32,
    "MAGMAR": 0x33,
    "MISSINGNO_34": 0x34,
    "ELECTABUZZ": 0x35,
    "MAGNETON": 0x36,
    "KOFFING": 0x37,
    "MISSINGNO_38": 0x38,
    "MANKEY": 0x39,
    "SEEL": 0x3A,
    "DIGLETT": 0x3B,
    "TAUROS": 0x3C,
    "MISSINGNO_3D": 0x3D,
    "MISSINGNO_3E": 0x3E,
    "MISSINGNO_3F": 0x3F,
    "FARFETCHD": 0x40,
    "VENONAT": 0x41,
    "DRAGONITE": 0x42,
    "MISSINGNO_43": 0x43,
    "MISSINGNO_44": 0x44,
    "MISSINGNO_45": 0x45,
    "DODUO": 0x46,
    "POLIWAG": 0x47,
    "JYNX": 0x48,
    "MOLTRES": 0x49,
    "ARTICUNO": 0x4A,
    "ZAPDOS": 0x4B,
    "DITTO": 0x4C,
    "MEOWTH": 0x4D,
    "KRABBY": 0x4E,
    "MISSINGNO_4F": 0x4F,
    "MISSINGNO_50": 0x50,
    "MISSINGNO_51": 0x51,
    "VULPIX": 0x52,
    "NINETALES": 0x53,
    "PIKACHU": 0x54,
    "RAICHU": 0x55,
    "MISSINGNO_56": 0x56,
    "MISSINGNO_57": 0x57,
    "DRATINI": 0x58,
    "DRAGONAIR": 0x59,
    "KABUTO": 0x5A,
    "KABUTOPS": 0x5B,
    "HORSEA": 0x5C,
    "SEADRA": 0x5D,
    "MISSINGNO_5E": 0x5E,
    "MISSINGNO_5F": 0x5F,
    "SANDSHREW": 0x60,
    "SANDSLASH": 0x61,
    "OMANYTE": 0x62,
    "OMASTAR": 0x63,
    "JIGGLYPUFF": 0x64,
    "WIGGLYTUFF": 0x65,
    "EEVEE": 0x66,
    "FLAREON": 0x67,
    "JOLTEON": 0x68,
    "VAPOREON": 0x69,
    "MACHOP": 0x6A,
    "ZUBAT": 0x6B,
    "EKANS": 0x6C,
    "PARAS": 0x6D,
    "POLIWHIRL": 0x6E,
    "POLIWRATH": 0x6F,
    "WEEDLE": 0x70,
    "KAKUNA": 0x71,
    "BEEDRILL": 0x72,
    "MISSINGNO_73": 0x73,
    "DODRIO": 0x74,
    "PRIMEAPE": 0x75,
    "DUGTRIO": 0x76,
    "VENOMOTH": 0x77,
    "DEWGONG": 0x78,
    "MISSINGNO_79": 0x79,
    "MISSINGNO_7A": 0x7A,
    "CATERPIE": 0x7B,
    "METAPOD": 0x7C,
    "BUTTERFREE": 0x7D,
    "MACHAMP": 0x7E,
    "MISSINGNO_7F": 0x7F,
    "GOLDUCK": 0x80,
    "HYPNO": 0x81,
    "GOLBAT": 0x82,
    "MEWTWO": 0x83,
    "SNORLAX": 0x84,
    "MAGIKARP": 0x85,
    "MISSINGNO_86": 0x86,
    "MISSINGNO_87": 0x87,
    "MUK": 0x88,
    "MISSINGNO_89": 0x89,
    "KINGLER": 0x8A,
    "CLOYSTER": 0x8B,
    "MISSINGNO_8C": 0x8C,
    "ELECTRODE": 0x8D,
    "CLEFABLE": 0x8E,
    "WEEZING": 0x8F,
    "PERSIAN": 0x90,
    "MAROWAK": 0x91,
    "MISSINGNO_92": 0x92,
    "HAUNTER": 0x93,
    "ABRA": 0x94,
    "ALAKAZAM": 0x95,
    "PIDGEOTTO": 0x96,
    "PIDGEOT": 0x97,
    "STARMIE": 0x98,
    "BULBASAUR": 0x99,
    "VENUSAUR": 0x9A,
    "TENTACRUEL": 0x9B,
    "MISSINGNO_9C": 0x9C,
    "GOLDEEN": 0x9D,
    "SEAKING": 0x9E,
    "MISSINGNO_9F": 0x9F,
    "MISSINGNO_A0": 0xA0,
    "MISSINGNO_A1": 0xA1,
    "MISSINGNO_A2": 0xA2,
    "PONYTA": 0xA3,
    "RAPIDASH": 0xA4,
    "RATTATA": 0xA5,
    "RATICATE": 0xA6,
    "NIDORINO": 0xA7,
    "NIDORINA": 0xA8,
    "GEODUDE": 0xA9,
    "PORYGON": 0xAA,
    "AERODACTYL": 0xAB,
    "MISSINGNO_AC": 0xAC,
    "MAGNEMITE": 0xAD,
    "MISSINGNO_AE": 0xAE,
    "MISSINGNO_AF": 0xAF,
    "CHARMANDER": 0xB0,
    "SQUIRTLE": 0xB1,
    "CHARMELEON": 0xB2,
    "WARTORTLE": 0xB3,
    "CHARIZARD": 0xB4,
    "MISSINGNO_B5": 0xB5,
    "FOSSIL_KABUTOPS": 0xB6,
    "FOSSIL_AERODACTYL": 0xB7,
    "MON_GHOST": 0xB8,
    "ODDISH": 0xB9,
    "GLOOM": 0xBA,
    "VILEPLUME": 0xBB,
    "BELLSPROUT": 0xBC,
    "WEEPINBELL": 0xBD,
    "VICTREEBEL": 0xBE,
}

STRENGTH_SOLUTIONS = {}

###################
# SEAFOAM ISLANDS #
###################

# Seafoam 1F Left
STRENGTH_SOLUTIONS[(63, 14, 22, 18, 11, 192)] = [
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "LEFT",
]
STRENGTH_SOLUTIONS[(63, 14, 22, 19, 10, 192)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 18, 11, 192)
]
STRENGTH_SOLUTIONS[(63, 14, 22, 18, 9, 192)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 19, 10, 192)
]
STRENGTH_SOLUTIONS[(63, 14, 22, 17, 10, 192)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 18, 9, 192)
]

# Seafoam 1F right
STRENGTH_SOLUTIONS[(63, 11, 30, 26, 8, 192)] = [
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]
STRENGTH_SOLUTIONS[(63, 11, 30, 27, 7, 192)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 26, 8, 192)
]
STRENGTH_SOLUTIONS[(63, 11, 30, 26, 6, 192)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 27, 7, 192)
]
STRENGTH_SOLUTIONS[(63, 11, 30, 25, 7, 192)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 26, 6, 192)
]

# Seafoam B1 left

STRENGTH_SOLUTIONS[(63, 10, 21, 16, 6, 159)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 21, 17, 5, 159)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 21, 16, 6, 159)
]
STRENGTH_SOLUTIONS[(63, 10, 21, 17, 7, 159)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 21, 16, 6, 159)
]

# Seafoam B1 right

STRENGTH_SOLUTIONS[(63, 10, 26, 21, 6, 159)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 26, 22, 5, 159)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 26, 21, 6, 159)
]
STRENGTH_SOLUTIONS[(63, 10, 26, 22, 7, 159)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 26, 21, 6, 159)
]

# Seafoam B2 left

STRENGTH_SOLUTIONS[(63, 10, 22, 17, 6, 160)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 22, 18, 5, 160)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 22, 17, 6, 160)
]
STRENGTH_SOLUTIONS[(63, 10, 22, 18, 7, 160)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 22, 17, 6, 160)
]

# Seafoam B2 right

STRENGTH_SOLUTIONS[(63, 10, 27, 24, 6, 160)] = ["LEFT"]
STRENGTH_SOLUTIONS[(63, 10, 27, 23, 7, 160)] = ["RIGHT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 27, 24, 6, 160)
]

# We skip seafoam b3 since that is for articuno
# TODO: Articuno

################
# VICTORY ROAD #
################

# 1F Switch 1
STRENGTH_SOLUTIONS[(63, 19, 9, 5, 14, 108)] = [
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "RIGHT",
    "DOWN",
]

STRENGTH_SOLUTIONS[(63, 19, 9, 4, 15, 108)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 19, 9, 5, 14, 108)
]
STRENGTH_SOLUTIONS[(63, 19, 9, 5, 16, 108)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 19, 9, 4, 15, 108)
]

# 2F Switch 1
STRENGTH_SOLUTIONS[(63, 18, 8, 5, 14, 194)] = [
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]

STRENGTH_SOLUTIONS[(63, 18, 8, 4, 13, 194)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 5, 14, 194)
]
STRENGTH_SOLUTIONS[(63, 18, 8, 3, 14, 194)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 4, 13, 194)
]
STRENGTH_SOLUTIONS[(63, 18, 8, 4, 15, 194)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 3, 14, 194)
]

# 3F Switch 3
STRENGTH_SOLUTIONS[(63, 19, 26, 22, 4, 198)] = [
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "RIGHT",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "RIGHT",
    "RIGHT",
]

STRENGTH_SOLUTIONS[(63, 19, 26, 23, 3, 198)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 22, 4, 198)
]
STRENGTH_SOLUTIONS[(63, 19, 26, 22, 2, 198)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 23, 3, 198)
]
STRENGTH_SOLUTIONS[(63, 19, 26, 21, 3, 198)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 22, 2, 198)
]

# 3F Boulder in hole
STRENGTH_SOLUTIONS[(63, 16, 17, 21, 15, 198)] = ["RIGHT", "RIGHT", "RIGHT"]
STRENGTH_SOLUTIONS[(63, 16, 17, 22, 16, 198)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 16, 17, 21, 15, 198)
]
STRENGTH_SOLUTIONS[(63, 16, 17, 22, 14, 198)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 16, 17, 21, 15, 198)
]


# 2F final switch
STRENGTH_SOLUTIONS[(63, 20, 27, 24, 16, 194)] = [
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]

STRENGTH_SOLUTIONS[(63, 20, 27, 23, 17, 194)] = ["RIGHT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 20, 27, 24, 16, 194)
]
STRENGTH_SOLUTIONS[(63, 20, 27, 22, 16, 194)] = ["DOWN", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 20, 27, 23, 17, 194)
]

from enum import Enum


class Tilesets(Enum):
    OVERWORLD = 0
    REDS_HOUSE_1 = 1
    MART = 2
    FOREST = 3
    REDS_HOUSE_2 = 4
    DOJO = 5
    POKECENTER = 6
    GYM = 7
    HOUSE = 8
    FOREST_GATE = 9
    MUSEUM = 10
    UNDERGROUND = 11
    GATE = 12
    SHIP = 13
    SHIP_PORT = 14
    CEMETERY = 15
    INTERIOR = 16
    CAVERN = 17
    LOBBY = 18
    MANSION = 19
    LAB = 20
    CLUB = 21
    FACILITY = 22
    PLATEAU = 23
    
TM_HM_MOVES = {
    5,  # Mega punch
    0xD,  # Razor wind
    0xE,  # Swords dance
    0x12,  # Whirlwind
    0x19,  # Mega kick
    0x5C,  # Toxic
    0x20,  # Horn drill
    0x22,  # Body slam
    0x24,  # Take down
    0x26,  # Double edge
    0x3D,  # Bubble beam
    0x37,  # Water gun
    0x3A,  # Ice beam
    0x3B,  # Blizzard
    0x3F,  # Hyper beam
    0x06,  # Pay day
    0x42,  # Submission
    0x44,  # Counter
    0x45,  # Seismic toss
    0x63,  # Rage
    0x48,  # Mega drain
    0x4C,  # Solar beam
    0x52,  # Dragon rage
    0x55,  # Thunderbolt
    0x57,  # Thunder
    0x59,  # Earthquake
    0x5A,  # Fissure
    0x5B,  # Dig
    0x5E,  # Psychic
    0x64,  # Teleport
    0x66,  # Mimic
    0x68,  # Double team
    0x73,  # Reflect
    0x75,  # Bide
    0x76,  # Metronome
    0x78,  # Selfdestruct
    0x79,  # Egg bomb
    0x7E,  # Fire blast
    0x81,  # Swift
    0x82,  # Skull bash
    0x87,  # Softboiled
    0x8A,  # Dream eater
    0x8F,  # Sky attack
    0x9C,  # Rest
    0x56,  # Thunder wave
    0x95,  # Psywave
    0x99,  # Explosion
    0x9D,  # Rock slide
    0xA1,  # Tri attack
    0xA4,  # Substitute
    0x0F,  # Cut
    0x13,  # Fly
    0x39,  # Surf
    0x46,  # Strength
    0x94,  # Flash
}

CUT_SPECIES_IDS = {
    SPECIES_IDS["BULBASAUR"],
    SPECIES_IDS["IVYSAUR"],
    SPECIES_IDS["VENUSAUR"],
    SPECIES_IDS["CHARMANDER"],
    SPECIES_IDS["CHARMELEON"],
    SPECIES_IDS["CHARIZARD"],
    SPECIES_IDS["BEEDRILL"],
    SPECIES_IDS["SANDSHREW"],
    SPECIES_IDS["SANDSLASH"],
    SPECIES_IDS["ODDISH"],
    SPECIES_IDS["GLOOM"],
    SPECIES_IDS["VILEPLUME"],
    SPECIES_IDS["PARAS"],
    SPECIES_IDS["PARASECT"],
    SPECIES_IDS["BELLSPROUT"],
    SPECIES_IDS["WEEPINBELL"],
    SPECIES_IDS["VICTREEBEL"],
    SPECIES_IDS["TENTACOOL"],
    SPECIES_IDS["TENTACRUEL"],
    SPECIES_IDS["FARFETCHD"],
    SPECIES_IDS["KRABBY"],
    SPECIES_IDS["KINGLER"],
    SPECIES_IDS["LICKITUNG"],
    SPECIES_IDS["TANGELA"],
    SPECIES_IDS["SCYTHER"],
    SPECIES_IDS["PINSIR"],
    SPECIES_IDS["MEW"],
}

SURF_SPECIES_IDS = {
    SPECIES_IDS["SQUIRTLE"],
    SPECIES_IDS["WARTORTLE"],
    SPECIES_IDS["BLASTOISE"],
    SPECIES_IDS["NIDOQUEEN"],
    SPECIES_IDS["NIDOKING"],
    SPECIES_IDS["PSYDUCK"],
    SPECIES_IDS["GOLDUCK"],
    SPECIES_IDS["POLIWAG"],
    SPECIES_IDS["POLIWHIRL"],
    SPECIES_IDS["POLIWRATH"],
    SPECIES_IDS["TENTACOOL"],
    SPECIES_IDS["TENTACRUEL"],
    SPECIES_IDS["SLOWPOKE"],
    SPECIES_IDS["SLOWBRO"],
    SPECIES_IDS["SEEL"],
    SPECIES_IDS["DEWGONG"],
    SPECIES_IDS["SHELLDER"],
    SPECIES_IDS["CLOYSTER"],
    SPECIES_IDS["KRABBY"],
    SPECIES_IDS["KINGLER"],
    SPECIES_IDS["LICKITUNG"],
    SPECIES_IDS["RHYDON"],
    SPECIES_IDS["KANGASKHAN"],
    SPECIES_IDS["HORSEA"],
    SPECIES_IDS["SEADRA"],
    SPECIES_IDS["GOLDEEN"],
    SPECIES_IDS["SEAKING"],
    SPECIES_IDS["STARYU"],
    SPECIES_IDS["STARMIE"],
    SPECIES_IDS["GYARADOS"],
    SPECIES_IDS["LAPRAS"],
    SPECIES_IDS["VAPOREON"],
    SPECIES_IDS["OMANYTE"],
    SPECIES_IDS["OMASTAR"],
    SPECIES_IDS["KABUTO"],
    SPECIES_IDS["KABUTOPS"],
    SPECIES_IDS["SNORLAX"],
    SPECIES_IDS["DRATINI"],
    SPECIES_IDS["DRAGONAIR"],
    SPECIES_IDS["DRAGONITE"],
    SPECIES_IDS["MEW"],
}

STRENGTH_SPECIES_IDS = {
    SPECIES_IDS["CHARMANDER"],
    SPECIES_IDS["CHARMELEON"],
    SPECIES_IDS["CHARIZARD"],
    SPECIES_IDS["SQUIRTLE"],
    SPECIES_IDS["WARTORTLE"],
    SPECIES_IDS["BLASTOISE"],
    SPECIES_IDS["EKANS"],
    SPECIES_IDS["ARBOK"],
    SPECIES_IDS["SANDSHREW"],
    SPECIES_IDS["SANDSLASH"],
    SPECIES_IDS["NIDOQUEEN"],
    SPECIES_IDS["NIDOKING"],
    SPECIES_IDS["CLEFAIRY"],
    SPECIES_IDS["CLEFABLE"],
    SPECIES_IDS["JIGGLYPUFF"],
    SPECIES_IDS["WIGGLYTUFF"],
    SPECIES_IDS["PSYDUCK"],
    SPECIES_IDS["GOLDUCK"],
    SPECIES_IDS["MANKEY"],
    SPECIES_IDS["PRIMEAPE"],
    SPECIES_IDS["POLIWHIRL"],
    SPECIES_IDS["POLIWRATH"],
    SPECIES_IDS["MACHOP"],
    SPECIES_IDS["MACHOKE"],
    SPECIES_IDS["MACHAMP"],
    SPECIES_IDS["GEODUDE"],
    SPECIES_IDS["GRAVELER"],
    SPECIES_IDS["GOLEM"],
    SPECIES_IDS["SLOWPOKE"],
    SPECIES_IDS["SLOWBRO"],
    SPECIES_IDS["SEEL"],
    SPECIES_IDS["DEWGONG"],
    SPECIES_IDS["GENGAR"],
    SPECIES_IDS["ONIX"],
    SPECIES_IDS["KRABBY"],
    SPECIES_IDS["KINGLER"],
    SPECIES_IDS["EXEGGUTOR"],
    SPECIES_IDS["CUBONE"],
    SPECIES_IDS["MAROWAK"],
    SPECIES_IDS["HITMONLEE"],
    SPECIES_IDS["HITMONCHAN"],
    SPECIES_IDS["LICKITUNG"],
    SPECIES_IDS["RHYHORN"],
    SPECIES_IDS["RHYDON"],
    SPECIES_IDS["CHANSEY"],
    SPECIES_IDS["KANGASKHAN"],
    SPECIES_IDS["ELECTABUZZ"],
    SPECIES_IDS["MAGMAR"],
    SPECIES_IDS["PINSIR"],
    SPECIES_IDS["TAUROS"],
    SPECIES_IDS["GYARADOS"],
    SPECIES_IDS["LAPRAS"],
    SPECIES_IDS["SNORLAX"],
    SPECIES_IDS["DRAGONITE"],
    SPECIES_IDS["MEWTWO"],
    SPECIES_IDS["MEW"],
}


class FieldMoves(Enum):
    CUT = 1
    FLY = 2
    SURF = 3
    SURF_2 = 4
    STRENGTH = 5
    FLASH = 6
    DIG = 7
    TELEPORT = 8
    SOFTBOILED = 9
    
class TmHmMoves(Enum):
    MEGA_PUNCH = (0x5,)
    RAZOR_WIND = 0xD
    SWORDS_DANCE = 0xE
    WHIRLWIND = 0x12
    MEGA_KICK = 0x19
    TOXIC = 0x5C
    HORN_DRILL = 0x20
    BODY_SLAM = 0x22
    TAKE_DOWN = 0x24
    DOUBLE_EDGE = 0x26
    BUBBLE_BEAM = 0x3D
    WATER_GUN = 0x37
    ICE_BEAM = 0x3A
    BLIZZARD = 0x3B
    HYPER_BEAM = 0x3F
    PAY_DAY = 0x06
    SUBMISSION = 0x42
    COUNTER = 0x44
    SEISMIC_TOSS = 0x45
    RAGE = 0x63
    MEGA_DRAIN = 0x48
    SOLAR_BEAM = 0x4C
    DRAGON_RAGE = 0x52
    THUNDERBOLT = 0x55
    THUNDER = 0x57
    EARTHQUAKE = 0x59
    FISSURE = 0x5A
    DIG = 0x5B
    PSYCHIC = 0x5E
    TELEPORT = 0x64
    MIMIC = 0x66
    DOUBLE_TEAM = 0x68
    REFLECT = 0x73
    BIDE = 0x75
    METRONOME = 0x76
    SELFDESTRUCT = 0x78
    EGG_BOMB = 0x79
    FIRE_BLAST = 0x7E
    SWIFT = 0x81
    SKULL_BASH = 0x82
    SOFTBOILED = 0x87
    DREAM_EATER = 0x8A
    SKY_ATTACK = 0x8F
    REST = 0x9C
    THUNDER_WAVE = 0x56
    PSYWAVE = 0x95
    EXPLOSION = 0x99
    ROCK_SLIDE = 0x9D
    TRI_ATTACK = 0xA1
    SUBSTITUTE = 0xA4
    CUT = 0x0F
    FLY = 0x13
    SURF = 0x39
    STRENGTH = 0x46
    FLASH = 0x94


class Species(Enum):
    RHYDON = 0x01
    KANGASKHAN = 0x02
    NIDORAN_M = 0x03
    CLEFAIRY = 0x04
    SPEAROW = 0x05
    VOLTORB = 0x06
    NIDOKING = 0x07
    SLOWBRO = 0x08
    IVYSAUR = 0x09
    EXEGGUTOR = 0x0A
    LICKITUNG = 0x0B
    EXEGGCUTE = 0x0C
    GRIMER = 0x0D
    GENGAR = 0x0E
    NIDORAN_F = 0x0F
    NIDOQUEEN = 0x10
    CUBONE = 0x11
    RHYHORN = 0x12
    LAPRAS = 0x13
    ARCANINE = 0x14
    MEW = 0x15
    GYARADOS = 0x16
    SHELLDER = 0x17
    TENTACOOL = 0x18
    GASTLY = 0x19
    SCYTHER = 0x1A
    STARYU = 0x1B
    BLASTOISE = 0x1C
    PINSIR = 0x1D
    TANGELA = 0x1E
    MISSINGNO_1F = 0x1F
    MISSINGNO_20 = 0x20
    GROWLITHE = 0x21
    ONIX = 0x22
    FEAROW = 0x23
    PIDGEY = 0x24
    SLOWPOKE = 0x25
    KADABRA = 0x26
    GRAVELER = 0x27
    CHANSEY = 0x28
    MACHOKE = 0x29
    MR_MIME = 0x2A
    HITMONLEE = 0x2B
    HITMONCHAN = 0x2C
    ARBOK = 0x2D
    PARASECT = 0x2E
    PSYDUCK = 0x2F
    DROWZEE = 0x30
    GOLEM = 0x31
    MISSINGNO_32 = 0x32
    MAGMAR = 0x33
    MISSINGNO_34 = 0x34
    ELECTABUZZ = 0x35
    MAGNETON = 0x36
    KOFFING = 0x37
    MISSINGNO_38 = 0x38
    MANKEY = 0x39
    SEEL = 0x3A
    DIGLETT = 0x3B
    TAUROS = 0x3C
    MISSINGNO_3D = 0x3D
    MISSINGNO_3E = 0x3E
    MISSINGNO_3F = 0x3F
    FARFETCHD = 0x40
    VENONAT = 0x41
    DRAGONITE = 0x42
    MISSINGNO_43 = 0x43
    MISSINGNO_44 = 0x44
    MISSINGNO_45 = 0x45
    DODUO = 0x46
    POLIWAG = 0x47
    JYNX = 0x48
    MOLTRES = 0x49
    ARTICUNO = 0x4A
    ZAPDOS = 0x4B
    DITTO = 0x4C
    MEOWTH = 0x4D
    KRABBY = 0x4E
    MISSINGNO_4F = 0x4F
    MISSINGNO_50 = 0x50
    MISSINGNO_51 = 0x51
    VULPIX = 0x52
    NINETALES = 0x53
    PIKACHU = 0x54
    RAICHU = 0x55
    MISSINGNO_56 = 0x56
    MISSINGNO_57 = 0x57
    DRATINI = 0x58
    DRAGONAIR = 0x59
    KABUTO = 0x5A
    KABUTOPS = 0x5B
    HORSEA = 0x5C
    SEADRA = 0x5D
    MISSINGNO_5E = 0x5E
    MISSINGNO_5F = 0x5F
    SANDSHREW = 0x60
    SANDSLASH = 0x61
    OMANYTE = 0x62
    OMASTAR = 0x63
    JIGGLYPUFF = 0x64
    WIGGLYTUFF = 0x65
    EEVEE = 0x66
    FLAREON = 0x67
    JOLTEON = 0x68
    VAPOREON = 0x69
    MACHOP = 0x6A
    ZUBAT = 0x6B
    EKANS = 0x6C
    PARAS = 0x6D
    POLIWHIRL = 0x6E
    POLIWRATH = 0x6F
    WEEDLE = 0x70
    KAKUNA = 0x71
    BEEDRILL = 0x72
    MISSINGNO_73 = 0x73
    DODRIO = 0x74
    PRIMEAPE = 0x75
    DUGTRIO = 0x76
    VENOMOTH = 0x77
    DEWGONG = 0x78
    MISSINGNO_79 = 0x79
    MISSINGNO_7A = 0x7A
    CATERPIE = 0x7B
    METAPOD = 0x7C
    BUTTERFREE = 0x7D
    MACHAMP = 0x7E
    MISSINGNO_7F = 0x7F
    GOLDUCK = 0x80
    HYPNO = 0x81
    GOLBAT = 0x82
    MEWTWO = 0x83
    SNORLAX = 0x84
    MAGIKARP = 0x85
    MISSINGNO_86 = 0x86
    MISSINGNO_87 = 0x87
    MUK = 0x88
    MISSINGNO_89 = 0x89
    KINGLER = 0x8A
    CLOYSTER = 0x8B
    MISSINGNO_8C = 0x8C
    ELECTRODE = 0x8D
    CLEFABLE = 0x8E
    WEEZING = 0x8F
    PERSIAN = 0x90
    MAROWAK = 0x91
    MISSINGNO_92 = 0x92
    HAUNTER = 0x93
    ABRA = 0x94
    ALAKAZAM = 0x95
    PIDGEOTTO = 0x96
    PIDGEOT = 0x97
    STARMIE = 0x98
    BULBASAUR = 0x99
    VENUSAUR = 0x9A
    TENTACRUEL = 0x9B
    MISSINGNO_9C = 0x9C
    GOLDEEN = 0x9D
    SEAKING = 0x9E
    MISSINGNO_9F = 0x9F
    MISSINGNO_A0 = 0xA0
    MISSINGNO_A1 = 0xA1
    MISSINGNO_A2 = 0xA2
    PONYTA = 0xA3
    RAPIDASH = 0xA4
    RATTATA = 0xA5
    RATICATE = 0xA6
    NIDORINO = 0xA7
    NIDORINA = 0xA8
    GEODUDE = 0xA9
    PORYGON = 0xAA
    AERODACTYL = 0xAB
    MISSINGNO_AC = 0xAC
    MAGNEMITE = 0xAD
    MISSINGNO_AE = 0xAE
    MISSINGNO_AF = 0xAF
    CHARMANDER = 0xB0
    SQUIRTLE = 0xB1
    CHARMELEON = 0xB2
    WARTORTLE = 0xB3
    CHARIZARD = 0xB4
    MISSINGNO_B5 = 0xB5
    FOSSIL_KABUTOPS = 0xB6
    FOSSIL_AERODACTYL = 0xB7
    MON_GHOST = 0xB8
    ODDISH = 0xB9
    GLOOM = 0xBA
    VILEPLUME = 0xBB
    BELLSPROUT = 0xBC
    WEEPINBELL = 0xBD
    VICTREEBEL = 0xBE

CUT_SPECIES_IDS = {
    Species.BULBASAUR.value,
    Species.IVYSAUR.value,
    Species.VENUSAUR.value,
    Species.CHARMANDER.value,
    Species.CHARMELEON.value,
    Species.CHARIZARD.value,
    Species.BEEDRILL.value,
    Species.SANDSHREW.value,
    Species.SANDSLASH.value,
    Species.ODDISH.value,
    Species.GLOOM.value,
    Species.VILEPLUME.value,
    Species.PARAS.value,
    Species.PARASECT.value,
    Species.BELLSPROUT.value,
    Species.WEEPINBELL.value,
    Species.VICTREEBEL.value,
    Species.TENTACOOL.value,
    Species.TENTACRUEL.value,
    Species.FARFETCHD.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.LICKITUNG.value,
    Species.TANGELA.value,
    Species.SCYTHER.value,
    Species.PINSIR.value,
    Species.MEW.value,
}

SURF_SPECIES_IDS = {
    Species.SQUIRTLE.value,
    Species.WARTORTLE.value,
    Species.BLASTOISE.value,
    Species.NIDOQUEEN.value,
    Species.NIDOKING.value,
    Species.PSYDUCK.value,
    Species.GOLDUCK.value,
    Species.POLIWAG.value,
    Species.POLIWHIRL.value,
    Species.POLIWRATH.value,
    Species.TENTACOOL.value,
    Species.TENTACRUEL.value,
    Species.SLOWPOKE.value,
    Species.SLOWBRO.value,
    Species.SEEL.value,
    Species.DEWGONG.value,
    Species.SHELLDER.value,
    Species.CLOYSTER.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.LICKITUNG.value,
    Species.RHYDON.value,
    Species.KANGASKHAN.value,
    Species.HORSEA.value,
    Species.SEADRA.value,
    Species.GOLDEEN.value,
    Species.SEAKING.value,
    Species.STARYU.value,
    Species.STARMIE.value,
    Species.GYARADOS.value,
    Species.LAPRAS.value,
    Species.VAPOREON.value,
    Species.OMANYTE.value,
    Species.OMASTAR.value,
    Species.KABUTO.value,
    Species.KABUTOPS.value,
    Species.SNORLAX.value,
    Species.DRATINI.value,
    Species.DRAGONAIR.value,
    Species.DRAGONITE.value,
    Species.MEW.value,
}

STRENGTH_SPECIES_IDS = {
    Species.CHARMANDER.value,
    Species.CHARMELEON.value,
    Species.CHARIZARD.value,
    Species.SQUIRTLE.value,
    Species.WARTORTLE.value,
    Species.BLASTOISE.value,
    Species.EKANS.value,
    Species.ARBOK.value,
    Species.SANDSHREW.value,
    Species.SANDSLASH.value,
    Species.NIDOQUEEN.value,
    Species.NIDOKING.value,
    Species.CLEFAIRY.value,
    Species.CLEFABLE.value,
    Species.JIGGLYPUFF.value,
    Species.WIGGLYTUFF.value,
    Species.PSYDUCK.value,
    Species.GOLDUCK.value,
    Species.MANKEY.value,
    Species.PRIMEAPE.value,
    Species.POLIWHIRL.value,
    Species.POLIWRATH.value,
    Species.MACHOP.value,
    Species.MACHOKE.value,
    Species.MACHAMP.value,
    Species.GEODUDE.value,
    Species.GRAVELER.value,
    Species.GOLEM.value,
    Species.SLOWPOKE.value,
    Species.SLOWBRO.value,
    Species.SEEL.value,
    Species.DEWGONG.value,
    Species.GENGAR.value,
    Species.ONIX.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.EXEGGUTOR.value,
    Species.CUBONE.value,
    Species.MAROWAK.value,
    Species.HITMONLEE.value,
    Species.HITMONCHAN.value,
    Species.LICKITUNG.value,
    Species.RHYHORN.value,
    Species.RHYDON.value,
    Species.CHANSEY.value,
    Species.KANGASKHAN.value,
    Species.ELECTABUZZ.value,
    Species.MAGMAR.value,
    Species.PINSIR.value,
    Species.TAUROS.value,
    Species.GYARADOS.value,
    Species.LAPRAS.value,
    Species.SNORLAX.value,
    Species.DRAGONITE.value,
    Species.MEWTWO.value,
    Species.MEW.value,
}

MAX_ITEM_CAPACITY = 20
# Starts at 0x1


class ItemsThatGuy(Enum):
    MASTER_BALL = 0x01
    ULTRA_BALL = 0x02
    GREAT_BALL = 0x03
    POKE_BALL = 0x04
    TOWN_MAP = 0x05
    BICYCLE = 0x06
    SURFBOARD = 0x07  #
    SAFARI_BALL = 0x08
    POKEDEX = 0x09
    MOON_STONE = 0x0A
    ANTIDOTE = 0x0B
    BURN_HEAL = 0x0C
    ICE_HEAL = 0x0D
    AWAKENING = 0x0E
    PARLYZ_HEAL = 0x0F
    FULL_RESTORE = 0x10
    MAX_POTION = 0x11
    HYPER_POTION = 0x12
    SUPER_POTION = 0x13
    POTION = 0x14
    BOULDERBADGE = 0x15
    CASCADEBADGE = 0x16
    SAFARI_BAIT = 0x15  # overload
    SAFARI_ROCK = 0x16  # overload
    THUNDERBADGE = 0x17
    RAINBOWBADGE = 0x18
    SOULBADGE = 0x19
    MARSHBADGE = 0x1A
    VOLCANOBADGE = 0x1B
    EARTHBADGE = 0x1C
    ESCAPE_ROPE = 0x1D
    REPEL = 0x1E
    OLD_AMBER = 0x1F
    FIRE_STONE = 0x20
    THUNDER_STONE = 0x21
    WATER_STONE = 0x22
    HP_UP = 0x23
    PROTEIN = 0x24
    IRON = 0x25
    CARBOS = 0x26
    CALCIUM = 0x27
    RARE_CANDY = 0x28
    DOME_FOSSIL = 0x29
    HELIX_FOSSIL = 0x2A
    SECRET_KEY = 0x2B
    UNUSED_ITEM = 0x2C  # "?????"
    BIKE_VOUCHER = 0x2D
    X_ACCURACY = 0x2E
    LEAF_STONE = 0x2F
    CARD_KEY = 0x30
    NUGGET = 0x31
    PP_UP_2 = 0x32
    POKE_DOLL = 0x33
    FULL_HEAL = 0x34
    REVIVE = 0x35
    MAX_REVIVE = 0x36
    GUARD_SPEC = 0x37
    SUPER_REPEL = 0x38
    MAX_REPEL = 0x39
    DIRE_HIT = 0x3A
    COIN = 0x3B
    FRESH_WATER = 0x3C
    SODA_POP = 0x3D
    LEMONADE = 0x3E
    S_S_TICKET = 0x3F
    GOLD_TEETH = 0x40
    X_ATTACK = 0x41
    X_DEFEND = 0x42
    X_SPEED = 0x43
    X_SPECIAL = 0x44
    COIN_CASE = 0x45
    OAKS_PARCEL = 0x46
    ITEMFINDER = 0x47
    SILPH_SCOPE = 0x48
    POKE_FLUTE = 0x49
    LIFT_KEY = 0x4A
    EXP_ALL = 0x4B
    OLD_ROD = 0x4C
    GOOD_ROD = 0x4D
    SUPER_ROD = 0x4E
    PP_UP = 0x4F
    ETHER = 0x50
    MAX_ETHER = 0x51
    ELIXER = 0x52
    MAX_ELIXER = 0x53
    FLOOR_B2F = 0x54
    FLOOR_B1F = 0x55
    FLOOR_1F = 0x56
    FLOOR_2F = 0x57
    FLOOR_3F = 0x58
    FLOOR_4F = 0x59
    FLOOR_5F = 0x5A
    FLOOR_6F = 0x5B
    FLOOR_7F = 0x5C
    FLOOR_8F = 0x5D
    FLOOR_9F = 0x5E
    FLOOR_10F = 0x5F
    FLOOR_11F = 0x60
    FLOOR_B4F = 0x61
    HM_01 = 0xC4
    HM_02 = 0xC5
    HM_03 = 0xC6
    HM_04 = 0xC7
    HM_05 = 0xC8
    TM_01 = 0xC9
    TM_02 = 0xCA
    TM_03 = 0xCB
    TM_04 = 0xCC
    TM_05 = 0xCD
    TM_06 = 0xCE
    TM_07 = 0xCF
    TM_08 = 0xD0
    TM_09 = 0xD1
    TM_10 = 0xD2
    TM_11 = 0xD3
    TM_12 = 0xD4
    TM_13 = 0xD5
    TM_14 = 0xD6
    TM_15 = 0xD7
    TM_16 = 0xD8
    TM_17 = 0xD9
    TM_18 = 0xDA
    TM_19 = 0xDB
    TM_20 = 0xDC
    TM_21 = 0xDD
    TM_22 = 0xDE
    TM_23 = 0xDF
    TM_24 = 0xE0
    TM_25 = 0xE1
    TM_26 = 0xE2
    TM_27 = 0xE3
    TM_28 = 0xE4
    TM_29 = 0xE5
    TM_30 = 0xE6
    TM_31 = 0xE7
    TM_32 = 0xE8
    TM_33 = 0xE9
    TM_34 = 0xEA
    TM_35 = 0xEB
    TM_36 = 0xEC
    TM_37 = 0xED
    TM_38 = 0xEE
    TM_39 = 0xEF
    TM_40 = 0xF0
    TM_41 = 0xF1
    TM_42 = 0xF2
    TM_43 = 0xF3
    TM_44 = 0xF4
    TM_45 = 0xF5
    TM_46 = 0xF6
    TM_47 = 0xF7
    TM_48 = 0xF8
    TM_49 = 0xF9
    TM_50 = 0xFA


KEY_ITEM_IDS = {
    ItemsThatGuy.TOWN_MAP.value,
    ItemsThatGuy.BICYCLE.value,
    ItemsThatGuy.SURFBOARD.value,
    ItemsThatGuy.SAFARI_BALL.value,
    ItemsThatGuy.POKEDEX.value,
    ItemsThatGuy.BOULDERBADGE.value,
    ItemsThatGuy.CASCADEBADGE.value,
    ItemsThatGuy.THUNDERBADGE.value,
    ItemsThatGuy.RAINBOWBADGE.value,
    ItemsThatGuy.SOULBADGE.value,
    ItemsThatGuy.MARSHBADGE.value,
    ItemsThatGuy.VOLCANOBADGE.value,
    ItemsThatGuy.EARTHBADGE.value,
    ItemsThatGuy.OLD_AMBER.value,
    ItemsThatGuy.DOME_FOSSIL.value,
    ItemsThatGuy.HELIX_FOSSIL.value,
    ItemsThatGuy.SECRET_KEY.value,
    # ItemsThatGuy.ITEM_2C.value,
    ItemsThatGuy.BIKE_VOUCHER.value,
    ItemsThatGuy.CARD_KEY.value,
    ItemsThatGuy.S_S_TICKET.value,
    ItemsThatGuy.GOLD_TEETH.value,
    ItemsThatGuy.COIN_CASE.value,
    ItemsThatGuy.OAKS_PARCEL.value,
    ItemsThatGuy.ITEMFINDER.value,
    ItemsThatGuy.SILPH_SCOPE.value,
    ItemsThatGuy.POKE_FLUTE.value,
    ItemsThatGuy.LIFT_KEY.value,
    ItemsThatGuy.OLD_ROD.value,
    ItemsThatGuy.GOOD_ROD.value,
    ItemsThatGuy.SUPER_ROD.value,
}

HM_ITEM_IDS = {
    ItemsThatGuy.HM_01.value,
    ItemsThatGuy.HM_02.value,
    ItemsThatGuy.HM_03.value,
    ItemsThatGuy.HM_04.value,
    ItemsThatGuy.HM_05.value,
}