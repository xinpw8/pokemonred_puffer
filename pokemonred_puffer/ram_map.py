import random
from . import data
import logging

# Configure logging
logging.basicConfig(
    filename="diagnostics.log",  # Name of the log file
    filemode="a",  # Append to the file
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Log level
)

# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309)
SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD886 # 0xD761
EVENTS_FLAGS_LENGTH = EVENT_FLAGS_END - EVENT_FLAGS_START
EXCLUDED_EVENTS = [
    (0xD747, 3),  # EVENT_HALL_OF_FAME_DEX_RATING
    (0xD790, 6),  # EVENT_SAFARI_GAME_OVER
    (0xD790, 7),  # EVENT_IN_SAFARI_ZONE
    (0xD751, 1),  # EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI
    (0xD751, 2),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_0
    (0xD751, 3),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_1
    (0xD751, 4),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_2
    (0xD751, 5),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_3
    (0xD751, 6),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_4
    (0xD751, 7),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_5
    (0xD752, 0),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_6
    (0xD752, 1),  # EVENT_BEAT_VIRIDIAN_GYM_TRAINER_7
]


MUSEUM_TICKET_ADDR = 0xD754
USED_CELL_SEPARATOR_ADDR = 0xD7F2
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349
# MAP_TEXT_POINTER_TABLE_NPC = 0xD36C - 0xD36D
TEXT_BOX_ARROW_BLINK = 0xC4F2
BATTLE_FLAG = 0xD057
SS_ANNE = 0xD803
IF_FONT_IS_LOADED = 0xCFC4  # text box is up
# get information for player
PLAYER_DIRECTION = 0xC109
PLAYER_Y = 0xC104
PLAYER_X = 0xC106
WNUMSPRITES = 0xD4E1
WNUMSIGNS = 0xD4B0
WCUTTILE = 0xCD4D  # $3d = tree tile; $52 = grass tile

HM_ITEM_IDS = set([0xC4, 0xC5, 0xC6, 0xC7, 0xC8])

# #Trainer Moves/PP counter if 00 then no move is present
# P1MOVES = [0xD173, 0xD174, 0xD175, 0xD176]
# P2MOVES = [0xD19F, 0xD1A0, 0xD1A1, 0xD1A2]
# P3MOVES = [0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE]
# P4MOVES = [0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA]
# P5MOVES = [0xD223, 0xD224, 0xD225, 0xD226]
# P6MOVES = [0xD24F, 0xD250, 0xD251, 0xD252]

# Moves 1-4 for Poke1, Poke2, Poke3, Poke4, Poke5, Poke6
MOVE1 = [0xD173, 0xD19F, 0xD1CB, 0xD1F7, 0xD223, 0xD24F]
MOVE2 = [0xD174, 0xD1A0, 0xD1CC, 0xD1F8, 0xD224, 0xD250]
MOVE3 = [0xD175, 0xD1A1, 0xD1CD, 0xD1F9, 0xD225, 0xD251]
MOVE4 = [0xD176, 0xD1A2, 0xD1CE, 0xD1FA, 0xD226, 0xD252]

MOVE1PP = [0xD188, 0xD1B4, 0xD1E0, 0xD20C, 0xD238, 0xD264]
MOVE2PP = [0xD189, 0xD1B5, 0xD1E1, 0xD20D, 0xD239, 0xD265]
MOVE3PP = [0xD18A, 0xD1B6, 0xD1E2, 0xD20E, 0xD23A, 0xD266]
MOVE4PP = [0xD18B, 0xD1B7, 0xD1E3, 0xD20F, 0xD23B, 0xD267]

items_dict = {
    1: {"decimal": 1, "hex": "0x01", "Item": "Master Ball"},
    2: {"decimal": 2, "hex": "0x02", "Item": "Ultra Ball"},
    3: {"decimal": 3, "hex": "0x03", "Item": "Great Ball"},
    4: {"decimal": 4, "hex": "0x04", "Item": "Poké Ball"},
    5: {"decimal": 5, "hex": "0x05", "Item": "Town Map"},
    6: {"decimal": 6, "hex": "0x06", "Item": "Bicycle"},
    7: {"decimal": 7, "hex": "0x07", "Item": "?????"},
    8: {"decimal": 8, "hex": "0x08", "Item": "Safari Ball"},
    9: {"decimal": 9, "hex": "0x09", "Item": "Pokédex"},
    10: {"decimal": 10, "hex": "0x0A", "Item": "Moon Stone"},
    11: {"decimal": 11, "hex": "0x0B", "Item": "Antidote"},
    12: {"decimal": 12, "hex": "0x0C", "Item": "Burn Heal"},
    13: {"decimal": 13, "hex": "0x0D", "Item": "Ice Heal"},
    14: {"decimal": 14, "hex": "0x0E", "Item": "Awakening"},
    15: {"decimal": 15, "hex": "0x0F", "Item": "Parlyz Heal"},
    16: {"decimal": 16, "hex": "0x10", "Item": "Full Restore"},
    17: {"decimal": 17, "hex": "0x11", "Item": "Max Potion"},
    18: {"decimal": 18, "hex": "0x12", "Item": "Hyper Potion"},
    19: {"decimal": 19, "hex": "0x13", "Item": "Super Potion"},
    20: {"decimal": 20, "hex": "0x14", "Item": "Potion"},
    21: {"decimal": 21, "hex": "0x15", "Item": "BoulderBadge"},
    22: {"decimal": 22, "hex": "0x16", "Item": "CascadeBadge"},
    23: {"decimal": 23, "hex": "0x17", "Item": "ThunderBadge"},
    24: {"decimal": 24, "hex": "0x18", "Item": "RainbowBadge"},
    25: {"decimal": 25, "hex": "0x19", "Item": "SoulBadge"},
    26: {"decimal": 26, "hex": "0x1A", "Item": "MarshBadge"},
    27: {"decimal": 27, "hex": "0x1B", "Item": "VolcanoBadge"},
    28: {"decimal": 28, "hex": "0x1C", "Item": "EarthBadge"},
    29: {"decimal": 29, "hex": "0x1D", "Item": "Escape Rope"},
    30: {"decimal": 30, "hex": "0x1E", "Item": "Repel"},
    31: {"decimal": 31, "hex": "0x1F", "Item": "Old Amber"},
    32: {"decimal": 32, "hex": "0x20", "Item": "Fire Stone"},
    33: {"decimal": 33, "hex": "0x21", "Item": "Thunderstone"},
    34: {"decimal": 34, "hex": "0x22", "Item": "Water Stone"},
    35: {"decimal": 35, "hex": "0x23", "Item": "HP Up"},
    36: {"decimal": 36, "hex": "0x24", "Item": "Protein"},
    37: {"decimal": 37, "hex": "0x25", "Item": "Iron"},
    38: {"decimal": 38, "hex": "0x26", "Item": "Carbos"},
    39: {"decimal": 39, "hex": "0x27", "Item": "Calcium"},
    40: {"decimal": 40, "hex": "0x28", "Item": "Rare Candy"},
    41: {"decimal": 41, "hex": "0x29", "Item": "Dome Fossil"},
    42: {"decimal": 42, "hex": "0x2A", "Item": "Helix Fossil"},
    43: {"decimal": 43, "hex": "0x2B", "Item": "Secret Key"},
    44: {"decimal": 44, "hex": "0x2C", "Item": "?????"},
    45: {"decimal": 45, "hex": "0x2D", "Item": "Bike Voucher"},
    46: {"decimal": 46, "hex": "0x2E", "Item": "X Accuracy"},
    47: {"decimal": 47, "hex": "0x2F", "Item": "Leaf Stone"},
    48: {"decimal": 48, "hex": "0x30", "Item": "Card Key"},
    49: {"decimal": 49, "hex": "0x31", "Item": "Nugget"},
    50: {"decimal": 50, "hex": "0x32", "Item": "PP Up*"},
    51: {"decimal": 51, "hex": "0x33", "Item": "Poké Doll"},
    52: {"decimal": 52, "hex": "0x34", "Item": "Full Heal"},
    53: {"decimal": 53, "hex": "0x35", "Item": "Revive"},
    54: {"decimal": 54, "hex": "0x36", "Item": "Max Revive"},
    55: {"decimal": 55, "hex": "0x37", "Item": "Guard Spec."},
    56: {"decimal": 56, "hex": "0x38", "Item": "Super Repel"},
    57: {"decimal": 57, "hex": "0x39", "Item": "Max Repel"},
    58: {"decimal": 58, "hex": "0x3A", "Item": "Dire Hit"},
    59: {"decimal": 59, "hex": "0x3B", "Item": "Coin"},
    60: {"decimal": 60, "hex": "0x3C", "Item": "Fresh Water"},
    61: {"decimal": 61, "hex": "0x3D", "Item": "Soda Pop"},
    62: {"decimal": 62, "hex": "0x3E", "Item": "Lemonade"},
    63: {"decimal": 63, "hex": "0x3F", "Item": "S.S. Ticket"},
    64: {"decimal": 64, "hex": "0x40", "Item": "Gold Teeth"},
    65: {"decimal": 65, "hex": "0x41", "Item": "X Attack"},
    66: {"decimal": 66, "hex": "0x42", "Item": "X Defend"},
    67: {"decimal": 67, "hex": "0x43", "Item": "X Speed"},
    68: {"decimal": 68, "hex": "0x44", "Item": "X Special"},
    69: {"decimal": 69, "hex": "0x45", "Item": "Coin Case"},
    70: {"decimal": 70, "hex": "0x46", "Item": "Oak's Parcel"},
    71: {"decimal": 71, "hex": "0x47", "Item": "Itemfinder"},
    72: {"decimal": 72, "hex": "0x48", "Item": "Silph Scope"},
    73: {"decimal": 73, "hex": "0x49", "Item": "Poké Flute"},
    74: {"decimal": 74, "hex": "0x4A", "Item": "Lift Key"},
    75: {"decimal": 75, "hex": "0x4B", "Item": "Exp. All"},
    76: {"decimal": 76, "hex": "0x4C", "Item": "Old Rod"},
    77: {"decimal": 77, "hex": "0x4D", "Item": "Good Rod"},
    78: {"decimal": 78, "hex": "0x4E", "Item": "Super Rod"},
    79: {"decimal": 79, "hex": "0x4F", "Item": "PP Up"},
    80: {"decimal": 80, "hex": "0x50", "Item": "Ether"},
    81: {"decimal": 81, "hex": "0x51", "Item": "Max Ether"},
    82: {"decimal": 82, "hex": "0x52", "Item": "Elixer"},
    83: {"decimal": 83, "hex": "0x53", "Item": "Max Elixer"},
    196: {"decimal": 196, "hex": "0xC4", "Item": "HM01"},
    197: {"decimal": 197, "hex": "0xC5", "Item": "HM02"},
    198: {"decimal": 198, "hex": "0xC6", "Item": "HM03"},
    199: {"decimal": 199, "hex": "0xC7", "Item": "HM04"},
    200: {"decimal": 200, "hex": "0xC8", "Item": "HM05"},
    201: {"decimal": 201, "hex": "0xC9", "Item": "TM01"},
    202: {"decimal": 202, "hex": "0xCA", "Item": "TM02"},
    203: {"decimal": 203, "hex": "0xCB", "Item": "TM03"},
    204: {"decimal": 204, "hex": "0xCC", "Item": "TM04"},
    205: {"decimal": 205, "hex": "0xCD", "Item": "TM05"},
    206: {"decimal": 206, "hex": "0xCE", "Item": "TM06"},
    207: {"decimal": 207, "hex": "0xCF", "Item": "TM07"},
    208: {"decimal": 208, "hex": "0xD0", "Item": "TM08"},
    209: {"decimal": 209, "hex": "0xD1", "Item": "TM09"},
    210: {"decimal": 210, "hex": "0xD2", "Item": "TM10"},
    211: {"decimal": 211, "hex": "0xD3", "Item": "TM11"},
    212: {"decimal": 212, "hex": "0xD4", "Item": "TM12"},
    213: {"decimal": 213, "hex": "0xD5", "Item": "TM13"},
    214: {"decimal": 214, "hex": "0xD6", "Item": "TM14"},
    215: {"decimal": 215, "hex": "0xD7", "Item": "TM15"},
    216: {"decimal": 216, "hex": "0xD8", "Item": "TM16"},
    217: {"decimal": 217, "hex": "0xD9", "Item": "TM17"},
    218: {"decimal": 218, "hex": "0xDA", "Item": "TM18"},
    219: {"decimal": 219, "hex": "0xDB", "Item": "TM19"},
    220: {"decimal": 220, "hex": "0xDC", "Item": "TM20"},
    221: {"decimal": 221, "hex": "0xDD", "Item": "TM21"},
    222: {"decimal": 222, "hex": "0xDE", "Item": "TM22"},
    223: {"decimal": 223, "hex": "0xDF", "Item": "TM23"},
    224: {"decimal": 224, "hex": "0xE0", "Item": "TM24"},
    225: {"decimal": 225, "hex": "0xE1", "Item": "TM25"},
    226: {"decimal": 226, "hex": "0xE2", "Item": "TM26"},
    227: {"decimal": 227, "hex": "0xE3", "Item": "TM27"},
    228: {"decimal": 228, "hex": "0xE4", "Item": "TM28"},
    229: {"decimal": 229, "hex": "0xE5", "Item": "TM29"},
    230: {"decimal": 230, "hex": "0xE6", "Item": "TM30"},
    231: {"decimal": 231, "hex": "0xE7", "Item": "TM31"},
    232: {"decimal": 232, "hex": "0xE8", "Item": "TM32"},
    233: {"decimal": 233, "hex": "0xE9", "Item": "TM33"},
    234: {"decimal": 234, "hex": "0xEA", "Item": "TM34"},
    235: {"decimal": 235, "hex": "0xEB", "Item": "TM35"},
    236: {"decimal": 236, "hex": "0xEC", "Item": "TM36"},
    237: {"decimal": 237, "hex": "0xED", "Item": "TM37"},
    238: {"decimal": 238, "hex": "0xEE", "Item": "TM38"},
    239: {"decimal": 239, "hex": "0xEF", "Item": "TM39"},
    240: {"decimal": 240, "hex": "0xF0", "Item": "TM40"},
    241: {"decimal": 241, "hex": "0xF1", "Item": "TM41"},
    242: {"decimal": 242, "hex": "0xF2", "Item": "TM42"},
    243: {"decimal": 243, "hex": "0xF3", "Item": "TM43"},
    244: {"decimal": 244, "hex": "0xF4", "Item": "TM44"},
    245: {"decimal": 245, "hex": "0xF5", "Item": "TM45"},
    246: {"decimal": 246, "hex": "0xF6", "Item": "TM46"},
    247: {"decimal": 247, "hex": "0xF7", "Item": "TM47"},
    248: {"decimal": 248, "hex": "0xF8", "Item": "TM48"},
    249: {"decimal": 249, "hex": "0xF9", "Item": "TM49"},
    250: {"decimal": 250, "hex": "0xFA", "Item": "TM50"},
    251: {"decimal": 251, "hex": "0xFB", "Item": "TM51"},
    252: {"decimal": 252, "hex": "0xFC", "Item": "TM52"},
    253: {"decimal": 253, "hex": "0xFD", "Item": "TM53"},
    254: {"decimal": 254, "hex": "0xFE", "Item": "TM54"},
    255: {"decimal": 255, "hex": "0xFF", "Item": "TM55"},
}


poke_dict = {
    3: {"hex": "3", "decimal": "3", "name": "Nidoran♂"},
    4: {"hex": "4", "decimal": "4", "name": "Clefairy"},
    5: {"hex": "5", "decimal": "5", "name": "Spearow"},
    9: {"hex": "9", "decimal": "9", "name": "Ivysaur"},
    15: {"hex": "F", "decimal": "15", "name": "Nidoran♀"},
    16: {"hex": "10", "decimal": "16", "name": "Nidoqueen"},
    28: {"hex": "1C", "decimal": "28", "name": "Blastoise"},
    35: {"hex": "23", "decimal": "35", "name": "Fearow"},
    36: {"hex": "24", "decimal": "36", "name": "Pidgey"},
    38: {"hex": "26", "decimal": "38", "name": "Kadabra"},
    39: {"hex": "27", "decimal": "39", "name": "Graveler"},
    46: {"hex": "2E", "decimal": "46", "name": "Parasect", "type": "Bug"},
    48: {"hex": "30", "decimal": "48", "name": "Drowzee"},
    49: {"hex": "31", "decimal": "49", "name": "Golem"},
    57: {"hex": "39", "decimal": "57", "name": "Mankey"},
    59: {"hex": "3B", "decimal": "59", "name": "Diglett"},
    70: {"hex": "46", "decimal": "70", "name": "Doduo"},
    84: {"hex": "54", "decimal": "84", "name": "Pikachu", "type": "Electric"},
    85: {"hex": "55", "decimal": "85", "name": "Raichu", "type": "Electric"},
    100: {"hex": "64", "decimal": "100", "name": "Jigglypuff"},
    101: {"hex": "65", "decimal": "101", "name": "Wigglytuff"},
    107: {"hex": "6B", "decimal": "107", "name": "Zubat"},
    108: {"hex": "6C", "decimal": "108", "name": "Ekans"},
    109: {"hex": "6D", "decimal": "109", "name": "Paras", "type": "Bug"},
    112: {"hex": "70", "decimal": "112", "name": "Weedle", "type": "Bug"},
    113: {"hex": "71", "decimal": "113", "name": "Kakuna", "type": "Bug"},
    114: {"hex": "72", "decimal": "114", "name": "Beedrill", "type": "Bug"},
    116: {"hex": "74", "decimal": "116", "name": "Dodrio"},
    117: {"hex": "75", "decimal": "117", "name": "Primeape"},
    118: {"hex": "76", "decimal": "118", "name": "Dugtrio"},
    123: {"hex": "7B", "decimal": "123", "name": "Caterpie", "type": "Bug"},
    124: {"hex": "7C", "decimal": "124", "name": "Metapod", "type": "Bug"},
    125: {"hex": "7D", "decimal": "125", "name": "Butterfree", "type": "Bug"},
    129: {"hex": "81", "decimal": "129", "name": "Hypno"},
    130: {"hex": "82", "decimal": "130", "name": "Golbat"},
    133: {"hex": "85", "decimal": "133", "name": "Magikarp"},
    142: {"hex": "8E", "decimal": "142", "name": "Clefable"},
    148: {"hex": "94", "decimal": "148", "name": "Abra"},
    149: {"hex": "95", "decimal": "149", "name": "Alakazam"},
    150: {"hex": "96", "decimal": "150", "name": "Pidgeotto"},
    151: {"hex": "97", "decimal": "151", "name": "Pidgeot"},
    153: {"hex": "99", "decimal": "153", "name": "Bulbasaur"},
    154: {"hex": "9A", "decimal": "154", "name": "Venusaur"},
    165: {"hex": "A5", "decimal": "165", "name": "Rattata"},
    166: {"hex": "A6", "decimal": "166", "name": "Raticate"},
    167: {"hex": "A7", "decimal": "167", "name": "Nidorino"},
    168: {"hex": "A8", "decimal": "168", "name": "Nidorina"},
    169: {"hex": "A9", "decimal": "169", "name": "Geodude"},
    176: {"hex": "B0", "decimal": "176", "name": "Charmander"},
    177: {"hex": "B1", "decimal": "177", "name": "Squirtle"},
    178: {"hex": "B2", "decimal": "178", "name": "Charmeleon"},
    179: {"hex": "B3", "decimal": "179", "name": "Wartortle"},
    180: {"hex": "B4", "decimal": "180", "name": "Charizard"},
    185: {"hex": "B9", "decimal": "185", "name": "Oddish"},
    186: {"hex": "BA", "decimal": "186", "name": "Gloom"},
    187: {"hex": "BB", "decimal": "187", "name": "Vileplume"},
}


wUnusedC000 = 0xC000
wSoundID = 0xC001
wMuteAudioAndPauseMusic = 0xC002
wDisableChannelOutputWhenSfxEnds = 0xC003
wStereoPanning = 0xC004
wSavedVolume = 0xC005
wChannelCommandPointers = 0xC006
wChannelReturnAddresses = 0xC016
wChannelSoundIDs = 0xC026
wChannelFlags1 = 0xC02E
wChannelFlags2 = 0xC036
wChannelDuties = 0xC03E
wChannelDutyCycles = 0xC046
wChannelVibratoDelayCounters = 0xC04E
wChannelVibratoExtents = 0xC056
wChannelVibratoRates = 0xC05E
wChannelFrequencyLowBytes = 0xC066
wChannelVibratoDelayCounterReloadValues = 0xC06E
wChannelPitchBendLengthModifiers = 0xC076
wChannelPitchBendFrequencySteps = 0xC07E
wChannelPitchBendFrequencyStepsFractionalPart = 0xC086
wChannelPitchBendCurrentFrequencyFractionalPart = 0xC08E
wChannelPitchBendCurrentFrequencyHighBytes = 0xC096
wChannelPitchBendCurrentFrequencyLowBytes = 0xC09E
wChannelPitchBendTargetFrequencyHighBytes = 0xC0A6
wChannelPitchBendTargetFrequencyLowBytes = 0xC0AE
wChannelNoteDelayCounters = 0xC0B6
wChannelLoopCounters = 0xC0BE
wChannelNoteSpeeds = 0xC0C6
wChannelNoteDelayCountersFractionalPart = 0xC0CE
wChannelOctaves = 0xC0D6
wChannelVolumes = 0xC0DE
wMusicTempo = 0xC0E8
wSfxTempo = 0xC0EA
wSfxHeaderPointer = 0xC0EC
wNewSoundID = 0xC0EE
wAudioROMBank = 0xC0EF
wAudioSavedROMBank = 0xC0F0
wFrequencyModifier = 0xC0F1
wTempoModifier = 0xC0F2
wSpriteStateData1 = 0xC100
wSpriteStateData2 = 0xC200
wOAMBuffer = 0xC300
wTileMap = 0xC3A0
wSerialPartyMonsPatchList = 0xC508
wTileMapBackup = 0xC508
wSerialEnemyMonsPatchList = 0xC5D0
wOverworldMap = 0xC6E8
wRedrawRowOrColumnSrcTiles = 0xCBFC
wTopMenuItemY = 0xCC24
wTopMenuItemX = 0xCC25
wCurrentMenuItem = 0xCC26
wTileBehindCursor = 0xCC27
wMaxMenuItem = 0xCC28
wMenuWatchedKeys = 0xCC29
wLastMenuItem = 0xCC2A
wPartyAndBillsPCSavedMenuItem = 0xCC2B
wBagSavedMenuItem = 0xCC2C
wBattleAndStartSavedMenuItem = 0xCC2D
wPlayerMoveListIndex = 0xCC2E
wPlayerMonNumber = 0xCC2F
wMenuCursorLocation = 0xCC30
wMenuJoypadPollCount = 0xCC34
wMenuItemToSwap = 0xCC35
wListScrollOffset = 0xCC36
wMenuWatchMovingOutOfBounds = 0xCC37
wTradeCenterPointerTableIndex = 0xCC38
wTextDest = 0xCC3A
wDoNotWaitForButtonPressAfterDisplayingText = 0xCC3C
wSerialSyncAndExchangeNybbleReceiveData = 0xCC3D
wSerialExchangeNybbleTempReceiveData = 0xCC3D
wLinkMenuSelectionReceiveBuffer = 0xCC3D
wSerialExchangeNybbleReceiveData = 0xCC3E
wSerialExchangeNybbleSendData = 0xCC42
wLinkMenuSelectionSendBuffer = 0xCC42
wLinkTimeoutCounter = 0xCC47
wUnknownSerialCounter = 0xCC47
wEnteringCableClub = 0xCC47
wWhichTradeMonSelectionMenu = 0xCC49
wMonDataLocation = 0xCC49
wMenuWrappingEnabled = 0xCC4A
wCheckFor180DegreeTurn = 0xCC4B
wMissableObjectIndex = 0xCC4D
wPredefID = 0xCC4E
wPredefRegisters = 0xCC4F
wTrainerHeaderFlagBit = 0xCC55
wNPCMovementScriptPointerTableNum = 0xCC57
wNPCMovementScriptBank = 0xCC58
wUnusedCC5B = 0xCC5B
wVermilionDockTileMapBuffer = 0xCC5B
wOaksAideRewardItemName = 0xCC5B
wDexRatingNumMonsSeen = 0xCC5B
wFilteredBagItems = 0xCC5B
wElevatorWarpMaps = 0xCC5B
wMonPartySpritesSavedOAM = 0xCC5B
wTrainerCardBlkPacket = 0xCC5B
wSlotMachineSevenAndBarModeChance = 0xCC5B
wHallOfFame = 0xCC5B
wBoostExpByExpAll = 0xCC5B
wAnimationType = 0xCC5B
wNPCMovementDirections = 0xCC5B
wDexRatingNumMonsOwned = 0xCC5C
wDexRatingText = 0xCC5D
wSlotMachineSavedROMBank = 0xCC5E
wAnimPalette = 0xCC79
wNPCMovementDirections2 = 0xCC97
wSwitchPartyMonTempBuffer = 0xCC97
wNumStepsToTake = 0xCCA1
wRLEByteCount = 0xCCD2
wAddedToParty = 0xCCD3
wSimulatedJoypadStatesEnd = 0xCCD3
wParentMenuItem = 0xCCD3
wCanEvolveFlags = 0xCCD3
wForceEvolution = 0xCCD4
wAILayer2Encouragement = 0xCCD5
wPlayerSubstituteHP = 0xCCD7
wEnemySubstituteHP = 0xCCD8
wTestBattlePlayerSelectedMove = 0xCCD9
wMoveMenuType = 0xCCDB
wPlayerSelectedMove = 0xCCDC
wEnemySelectedMove = 0xCCDD
wLinkBattleRandomNumberListIndex = 0xCCDE
wAICount = 0xCCDF
wEnemyMoveListIndex = 0xCCE2
wLastSwitchInEnemyMonHP = 0xCCE3
wTotalPayDayMoney = 0xCCE5
wSafariEscapeFactor = 0xCCE8
wSafariBaitFactor = 0xCCE9
wTransformedEnemyMonOriginalDVs = 0xCCEB
wInHandlePlayerMonFainted = 0xCCF0
wPartyFoughtCurrentEnemyFlags = 0xCCF5
wLowHealthAlarmDisabled = 0xCCF6
wPlayerMonMinimized = 0xCCF7
wLuckySlotHiddenObjectIndex = 0xCD05
wEnemyNumHits = 0xCD05
wEnemyBideAccumulatedDamage = 0xCD05
wInGameTradeGiveMonSpecies = 0xCD0F
wPlayerMonUnmodifiedLevel = 0xCD0F
wInGameTradeTextPointerTablePointer = 0xCD10
wPlayerMonUnmodifiedMaxHP = 0xCD10
wInGameTradeTextPointerTableIndex = 0xCD12
wPlayerMonUnmodifiedAttack = 0xCD12
wInGameTradeGiveMonName = 0xCD13
wPlayerMonUnmodifiedDefense = 0xCD14
wPlayerMonUnmodifiedSpeed = 0xCD16
wPlayerMonUnmodifiedSpecial = 0xCD18
wPlayerMonAttackMod = 0xCD1A
wPlayerMonDefenseMod = 0xCD1B
wPlayerMonSpeedMod = 0xCD1C
wPlayerMonSpecialMod = 0xCD1D
wInGameTradeReceiveMonName = 0xCD1E
wPlayerMonAccuracyMod = 0xCD1E
wPlayerMonEvasionMod = 0xCD1F
wEnemyMonUnmodifiedLevel = 0xCD23
wEnemyMonUnmodifiedMaxHP = 0xCD24
wEnemyMonUnmodifiedAttack = 0xCD26
wEnemyMonUnmodifiedDefense = 0xCD28
wInGameTradeMonNick = 0xCD29
wEnemyMonUnmodifiedSpeed = 0xCD2A
wEnemyMonUnmodifiedSpecial = 0xCD2C
wEngagedTrainerClass = 0xCD2D
wEngagedTrainerSet = 0xCD2E
wEnemyMonAttackMod = 0xCD2E
wEnemyMonDefenseMod = 0xCD2F
wEnemyMonSpeedMod = 0xCD30
wEnemyMonSpecialMod = 0xCD31
wEnemyMonAccuracyMod = 0xCD32
wEnemyMonEvasionMod = 0xCD33
wNPCMovementDirections2Index = 0xCD37
wUnusedCD37 = 0xCD37
wFilteredBagItemsCount = 0xCD37
wSimulatedJoypadStatesIndex = 0xCD38
wWastedByteCD39 = 0xCD39
wWastedByteCD3A = 0xCD3A
wOverrideSimulatedJoypadStatesMask = 0xCD3B
wFallingObjectsMovementData = 0xCD3D
wSavedY = 0xCD3D
wTempSCX = 0xCD3D
wBattleTransitionCircleScreenQuadrantY = 0xCD3D
wBattleTransitionCopyTilesOffset = 0xCD3D
wInwardSpiralUpdateScreenCounter = 0xCD3D
wHoFTeamIndex = 0xCD3D
wSSAnneSmokeDriftAmount = 0xCD3D
wRivalStarterTemp = 0xCD3D
wBoxMonCounts = 0xCD3D
wDexMaxSeenMon = 0xCD3D
wPPRestoreItem = 0xCD3D
wWereAnyMonsAsleep = 0xCD3D
wCanPlaySlots = 0xCD3D
wNumShakes = 0xCD3D
wDayCareStartLevel = 0xCD3D
wWhichBadge = 0xCD3D
wPriceTemp = 0xCD3D
wTitleMonSpecies = 0xCD3D
wPlayerCharacterOAMTile = 0xCD3D
wMoveDownSmallStarsOAMCount = 0xCD3D
wChargeMoveNum = 0xCD3D
wCoordIndex = 0xCD3D
wOptionsTextSpeedCursorX = 0xCD3D
wBoxNumString = 0xCD3D
wTrainerInfoTextBoxWidthPlus1 = 0xCD3D
wSwappedMenuItem = 0xCD3D
wHoFMonSpecies = 0xCD3D
wFieldMoves = 0xCD3D
wBadgeNumberTile = 0xCD3D
wRodResponse = 0xCD3D
wWhichTownMapLocation = 0xCD3D
wStoppingWhichSlotMachineWheel = 0xCD3D
wTradedPlayerMonSpecies = 0xCD3D
wTradingWhichPlayerMon = 0xCD3D
wChangeBoxSavedMapTextPointer = 0xCD3D
wFlyAnimUsingCoordList = 0xCD3D
wPlayerSpinInPlaceAnimFrameDelay = 0xCD3D
wPlayerSpinWhileMovingUpOrDownAnimDeltaY = 0xCD3D
wHiddenObjectFunctionArgument = 0xCD3D
wWhichTrade = 0xCD3D
wTrainerSpriteOffset = 0xCD3D
wUnusedCD3D = 0xCD3D
wHUDPokeballGfxOffsetX = 0xCD3E
wBattleTransitionCircleScreenQuadrantX = 0xCD3E
wSSAnneSmokeX = 0xCD3E
wRivalStarterBallSpriteIndex = 0xCD3E
wDayCareNumLevelsGrown = 0xCD3E
wOptionsBattleAnimCursorX = 0xCD3E
wTrainerInfoTextBoxWidth = 0xCD3E
wHoFPartyMonIndex = 0xCD3E
wNumCreditsMonsDisplayed = 0xCD3E
wBadgeNameTile = 0xCD3E
wFlyLocationsList = 0xCD3E
wSlotMachineWheel1Offset = 0xCD3E
wTradedEnemyMonSpecies = 0xCD3E
wTradingWhichEnemyMon = 0xCD3E
wFlyAnimCounter = 0xCD3E
wPlayerSpinInPlaceAnimFrameDelayDelta = 0xCD3E
wPlayerSpinWhileMovingUpOrDownAnimMaxY = 0xCD3E
wHiddenObjectFunctionRomBank = 0xCD3E
wTrainerEngageDistance = 0xCD3E
wHUDGraphicsTiles = 0xCD3F
wDayCareTotalCost = 0xCD3F
wJigglypuffFacingDirections = 0xCD3F
wOptionsBattleStyleCursorX = 0xCD3F
wTrainerInfoTextBoxNextRowOffset = 0xCD3F
wHoFMonLevel = 0xCD3F
wBadgeOrFaceTiles = 0xCD3F
wSlotMachineWheel2Offset = 0xCD3F
wNameOfPlayerMonToBeTraded = 0xCD3F
wFlyAnimBirdSpriteImageIndex = 0xCD3F
wPlayerSpinInPlaceAnimFrameDelayEndValue = 0xCD3F
wPlayerSpinWhileMovingUpOrDownAnimFrameDelay = 0xCD3F
wHiddenObjectIndex = 0xCD3F
wTrainerFacingDirection = 0xCD3F
wHoFMonOrPlayer = 0xCD40
wSlotMachineWheel3Offset = 0xCD40
wPlayerSpinInPlaceAnimSoundID = 0xCD40
wHiddenObjectY = 0xCD40
wTrainerScreenY = 0xCD40
wUnusedCD40 = 0xCD40
wDayCarePerLevelCost = 0xCD41
wHoFTeamIndex2 = 0xCD41
wHiddenItemOrCoinsIndex = 0xCD41
wTradedPlayerMonOT = 0xCD41
wHiddenObjectX = 0xCD41
wSlotMachineWinningSymbol = 0xCD41
wNumFieldMoves = 0xCD41
wSlotMachineWheel1BottomTile = 0xCD41
wTrainerScreenX = 0xCD41
wHoFTeamNo = 0xCD42
wSlotMachineWheel1MiddleTile = 0xCD42
wFieldMovesLeftmostXCoord = 0xCD42
wLastFieldMoveID = 0xCD43
wSlotMachineWheel1TopTile = 0xCD43
wSlotMachineWheel2BottomTile = 0xCD44
wSlotMachineWheel2MiddleTile = 0xCD45
wTempCoins1 = 0xCD46
wSlotMachineWheel2TopTile = 0xCD46
wBattleTransitionSpiralDirection = 0xCD47
wSlotMachineWheel3BottomTile = 0xCD47
wSlotMachineWheel3MiddleTile = 0xCD48
wFacingDirectionList = 0xCD48
wSlotMachineWheel3TopTile = 0xCD49
wTempCoins2 = 0xCD4A
wPayoutCoins = 0xCD4A
wTradedPlayerMonOTID = 0xCD4C
wSlotMachineFlags = 0xCD4C
wSlotMachineWheel1SlipCounter = 0xCD4D
wCutTile = 0xCD4D
wSlotMachineWheel2SlipCounter = 0xCD4E
wTradedEnemyMonOT = 0xCD4E
wSavedPlayerScreenY = 0xCD4F
wSlotMachineRerollCounter = 0xCD4F
wEmotionBubbleSpriteIndex = 0xCD4F
wWhichEmotionBubble = 0xCD50
wSlotMachineBet = 0xCD50
wSavedPlayerFacingDirection = 0xCD50
wWhichAnimationOffsets = 0xCD50
wTradedEnemyMonOTID = 0xCD59
wStandingOnWarpPadOrHole = 0xCD5B
wOAMBaseTile = 0xCD5B
wGymTrashCanIndex = 0xCD5B
wSymmetricSpriteOAMAttributes = 0xCD5C
wMonPartySpriteSpecies = 0xCD5D
wLeftGBMonSpecies = 0xCD5E
wRightGBMonSpecies = 0xCD5F
wFlags_0xcd60 = 0xCD60
wActionResultOrTookBattleTurn = 0xCD6A
wJoyIgnore = 0xCD6B
wDownscaledMonSize = 0xCD6C
wNumMovesMinusOne = 0xCD6C
wStatusScreenCurrentPP = 0xCD71
wNormalMaxPPList = 0xCD78
wSerialOtherGameboyRandomNumberListBlock = 0xCD81
wTileMapBackup2 = 0xCD81
wNamingScreenNameLength = 0xCEE9
wEvoOldSpecies = 0xCEE9
wBuffer = 0xCEE9
wTownMapCoords = 0xCEE9
wLearningMovesFromDayCare = 0xCEE9
wChangeMonPicEnemyTurnSpecies = 0xCEE9
wHPBarMaxHP = 0xCEE9
wNamingScreenSubmitName = 0xCEEA
wChangeMonPicPlayerTurnSpecies = 0xCEEA
wEvoNewSpecies = 0xCEEA
wAlphabetCase = 0xCEEB
wEvoMonTileOffset = 0xCEEB
wHPBarOldHP = 0xCEEB
wEvoCancelled = 0xCEEC
wNamingScreenLetter = 0xCEED
wHPBarNewHP = 0xCEED
wHPBarDelta = 0xCEEF
wHPBarTempHP = 0xCEF0
wHPBarHPDifference = 0xCEFD
wAIItem = 0xCF05
wUsedItemOnWhichPokemon = 0xCF05
wAnimSoundID = 0xCF07
wBankswitchHomeSavedROMBank = 0xCF08
wBankswitchHomeTemp = 0xCF09
wBoughtOrSoldItemInMart = 0xCF0A
wBattleResult = 0xCF0B
wAutoTextBoxDrawingControl = 0xCF0C
wTilePlayerStandingOn = 0xCF0E
wNPCMovementScriptFunctionNum = 0xCF10
wTextPredefFlag = 0xCF11
wPredefParentBank = 0xCF12
wCurSpriteMovement2 = 0xCF14
wNPCMovementScriptSpriteOffset = 0xCF17
wScriptedNPCWalkCounter = 0xCF18
wGBC = 0xCF1A
wOnSGB = 0xCF1B
wDefaultPaletteCommand = 0xCF1C
wPlayerHPBarColor = 0xCF1D
wWholeScreenPaletteMonSpecies = 0xCF1D
wEnemyHPBarColor = 0xCF1E
wPartyMenuHPBarColors = 0xCF1F
wStatusScreenHPBarColor = 0xCF25
wCopyingSGBTileData = 0xCF2D
wWhichPartyMenuHPBar = 0xCF2D
wPalPacket = 0xCF2D
wPartyMenuBlkPacket = 0xCF2E
wExpAmountGained = 0xCF4B
wGainBoostedExp = 0xCF4D
wGymCityName = 0xCF5F
wGymLeaderName = 0xCF70
wItemList = 0xCF7B
wListPointer = 0xCF8B
wUnusedCF8D = 0xCF8D
wItemPrices = 0xCF8F
wWhichPokemon = 0xCF92
wPrintItemPrices = 0xCF93
wHPBarType = 0xCF94
wListMenuID = 0xCF94
wRemoveMonFromBox = 0xCF95
wMoveMonType = 0xCF95
wItemQuantity = 0xCF96
wMaxItemQuantity = 0xCF97
wFontLoaded = 0xCFC4
wWalkCounter = 0xCFC5
wTileInFrontOfPlayer = 0xCFC6
wAudioFadeOutControl = 0xCFC7
wAudioFadeOutCounterReloadValue = 0xCFC8
wAudioFadeOutCounter = 0xCFC9
wLastMusicSoundID = 0xCFCA
wUpdateSpritesEnabled = 0xCFCB
wEnemyMoveNum = 0xCFCC
wEnemyMoveEffect = 0xCFCD
wEnemyMovePower = 0xCFCE
wEnemyMoveType = 0xCFCF
wEnemyMoveAccuracy = 0xCFD0
wEnemyMoveMaxPP = 0xCFD1
wPlayerMoveNum = 0xCFD2
wPlayerMoveEffect = 0xCFD3
wPlayerMovePower = 0xCFD4
wPlayerMoveType = 0xCFD5
wPlayerMoveAccuracy = 0xCFD6
wPlayerMoveMaxPP = 0xCFD7
wEnemyMonSpecies2 = 0xCFD8
wBattleMonSpecies2 = 0xCFD9
wTrainerClass = 0xD031
wTrainerPicPointer = 0xD033
wTempMoveNameBuffer = 0xD036
wLearnMoveMonName = 0xD036
wTrainerBaseMoney = 0xD046
wMissableObjectCounter = 0xD048
wTrainerName = 0xD04A
wIsInBattle = 0xD057
wPartyGainExpFlags = 0xD058
wCurOpponent = 0xD059
wBattleType = 0xD05A
wDamageMultipliers = 0xD05B
wLoneAttackNo = 0xD05C
wGymLeaderNo = 0xD05C
wTrainerNo = 0xD05D
wCriticalHitOrOHKO = 0xD05E
wMoveMissed = 0xD05F
wPlayerStatsToDouble = 0xD060
wPlayerStatsToHalve = 0xD061
wPlayerBattleStatus1 = 0xD062
wPlayerBattleStatus2 = 0xD063
wPlayerBattleStatus3 = 0xD064
wEnemyStatsToDouble = 0xD065
wEnemyStatsToHalve = 0xD066
wEnemyBattleStatus1 = 0xD067
wEnemyBattleStatus2 = 0xD068
wEnemyBattleStatus3 = 0xD069
wPlayerConfusedCounter = 0xD06B
wPlayerToxicCounter = 0xD06C
wPlayerDisabledMove = 0xD06D
wEnemyNumAttacksLeft = 0xD06F
wEnemyConfusedCounter = 0xD070
wEnemyToxicCounter = 0xD071
wEnemyDisabledMove = 0xD072
wPlayerNumHits = 0xD074
wPlayerBideAccumulatedDamage = 0xD074
wUnknownSerialCounter2 = 0xD075
wAmountMoneyWon = 0xD079
wObjectToHide = 0xD079
wObjectToShow = 0xD07A
wDefaultMap = 0xD07C
wMenuItemOffset = 0xD07C
wAnimationID = 0xD07C
wNamingScreenType = 0xD07D
wPartyMenuTypeOrMessageID = 0xD07D
wTempTilesetNumTiles = 0xD07D
wSavedListScrollOffset = 0xD07E
wBaseCoordX = 0xD081
wBaseCoordY = 0xD082
wFBTileCounter = 0xD084
wMovingBGTilesCounter2 = 0xD085
wSubAnimFrameDelay = 0xD086
wSubAnimCounter = 0xD087
wSaveFileStatus = 0xD088
wNumFBTiles = 0xD089
wFlashScreenLongCounter = 0xD08A
wSpiralBallsBaseY = 0xD08A
wFallingObjectMovementByte = 0xD08A
wNumShootingBalls = 0xD08A
wTradedMonMovingRight = 0xD08A
wOptionsInitialized = 0xD08A
wNewSlotMachineBallTile = 0xD08A
wCoordAdjustmentAmount = 0xD08A
wUnusedD08A = 0xD08A
wSpiralBallsBaseX = 0xD08B
wNumFallingObjects = 0xD08B
wSlideMonDelay = 0xD08B
wAnimCounter = 0xD08B
wSubAnimTransform = 0xD08B
wEndBattleWinTextPointer = 0xD08C
wEndBattleLoseTextPointer = 0xD08E
wEndBattleTextRomBank = 0xD092
wSubAnimAddrPtr = 0xD094
wSlotMachineAllowMatchesCounter = 0xD096
wSubAnimSubEntryAddr = 0xD096
wOutwardSpiralTileMapPointer = 0xD09A
wPartyMenuAnimMonEnabled = 0xD09B
wTownMapSpriteBlinkingEnabled = 0xD09B
wUnusedD09B = 0xD09B
wFBDestAddr = 0xD09C
wFBMode = 0xD09E
wLinkCableAnimBulgeToggle = 0xD09F
wIntroNidorinoBaseTile = 0xD09F
wOutwardSpiralCurrentDirection = 0xD09F
wDropletTile = 0xD09F
wNewTileBlockID = 0xD09F
wWhichBattleAnimTileset = 0xD09F
wSquishMonCurrentDirection = 0xD09F
wSlideMonUpBottomRowLeftTile = 0xD09F
wSpriteCurPosX = 0xD0A1
wSpriteCurPosY = 0xD0A2
wSpriteWidth = 0xD0A3
wSpriteHeight = 0xD0A4
wSpriteInputCurByte = 0xD0A5
wSpriteInputBitCounter = 0xD0A6
wSpriteOutputBitOffset = 0xD0A7
wSpriteLoadFlags = 0xD0A8
wSpriteUnpackMode = 0xD0A9
wSpriteFlipped = 0xD0AA
wSpriteInputPtr = 0xD0AB
wSpriteOutputPtr = 0xD0AD
wSpriteOutputPtrCached = 0xD0AF
wSpriteDecodeTable0Ptr = 0xD0B1
wSpriteDecodeTable1Ptr = 0xD0B3
wNameListType = 0xD0B6
wPredefBank = 0xD0B7
wMonHeader = 0xD0B8
wMonHIndex = 0xD0B8
wMonHBaseStats = 0xD0B9
wMonHBaseHP = 0xD0B9
wMonHBaseAttack = 0xD0BA
wMonHBaseDefense = 0xD0BB
wMonHBaseSpeed = 0xD0BC
wMonHBaseSpecial = 0xD0BD
wMonHTypes = 0xD0BE
wMonHType1 = 0xD0BE
wMonHType2 = 0xD0BF
wMonHCatchRate = 0xD0C0
wMonHBaseEXP = 0xD0C1
wMonHSpriteDim = 0xD0C2
wMonHFrontSprite = 0xD0C3
wMonHBackSprite = 0xD0C5
wMonHMoves = 0xD0C7
wMonHGrowthRate = 0xD0CB
wMonHLearnset = 0xD0CC
wSavedTilesetType = 0xD0D4
wDamage = 0xD0D7
wRepelRemainingSteps = 0xD0DB
wMoves = 0xD0DC
wMoveNum = 0xD0E0
wMovesString = 0xD0E1
wUnusedD119 = 0xD119
wWalkBikeSurfStateCopy = 0xD11A
wInitListType = 0xD11B
wCapturedMonSpecies = 0xD11C
wFirstMonsNotOutYet = 0xD11D
wPokeBallCaptureCalcTemp = 0xD11E
wPokeBallAnimData = 0xD11E
wUsingPPUp = 0xD11E
wMaxPP = 0xD11E
wCalculateWhoseStats = 0xD11E
wTypeEffectiveness = 0xD11E
wMoveType = 0xD11E
wNumSetBits = 0xD11E
wForcePlayerToChooseMon = 0xD11F
wEvolutionOccurred = 0xD121
wVBlankSavedROMBank = 0xD122
wIsKeyItem = 0xD124
wTextBoxID = 0xD125
wCurEnemyLVL = 0xD127
wItemListPointer = 0xD128
wLinkState = 0xD12B
wTwoOptionMenuID = 0xD12C
wChosenMenuItem = 0xD12D
wOutOfBattleBlackout = 0xD12D
wMenuExitMethod = 0xD12E
wDungeonWarpDataEntrySize = 0xD12F
wWhichPewterGuy = 0xD12F
wWhichPrizeWindow = 0xD12F
wGymGateTileBlock = 0xD12F
wSavedSpriteScreenY = 0xD130
wSavedSpriteScreenX = 0xD131
wSavedSpriteMapY = 0xD132
wSavedSpriteMapX = 0xD133
wWhichPrize = 0xD139
wIgnoreInputCounter = 0xD13A
wStepCounter = 0xD13B
wNumberOfNoRandomBattleStepsLeft = 0xD13C
wPrize1 = 0xD13D
wPrize2 = 0xD13E
wPrize3 = 0xD13F
wSerialRandomNumberListBlock = 0xD141
wPrize1Price = 0xD141
wPrize2Price = 0xD143
wPrize3Price = 0xD145
wLinkBattleRandomNumberList = 0xD148
wSerialPlayerDataBlock = 0xD152
wPseudoItemID = 0xD152
wUnusedD153 = 0xD153
wEvoStoneItemID = 0xD156
wSavedNPCMovementDirections2Index = 0xD157
wPlayerName = 0xD158
wPokedexOwned = 0xD2F7
wPokedexSeen = 0xD30A
wNumBagItems = 0xD31D
wBagItems = 0xD31E
wPlayerMoney = 0xD347
wRivalName = 0xD34A
wOptions = 0xD355
wObtainedBadges = 0xD356
wLetterPrintingDelayFlags = 0xD358
wPlayerID = 0xD359
wMapMusicSoundID = 0xD35B
wMapMusicROMBank = 0xD35C
wMapPalOffset = 0xD35D
wCurMap = 0xD35E
wCurrentTileBlockMapViewPointer = 0xD35F
wYCoord = 0xD361
wXCoord = 0xD362
wYBlockCoord = 0xD363
wXBlockCoord = 0xD364
wLastMap = 0xD365
wUnusedD366 = 0xD366
wCurMapTileset = 0xD367
wCurMapHeight = 0xD368
wCurMapWidth = 0xD369
wMapDataPtr = 0xD36A
wMapTextPtr = 0xD36C
wMapScriptPtr = 0xD36E
wMapConnections = 0xD370
wMapConn1Ptr = 0xD371
wNorthConnectionStripSrc = 0xD372
wNorthConnectionStripDest = 0xD374
wNorthConnectionStripWidth = 0xD376
wNorthConnectedMapWidth = 0xD377
wNorthConnectedMapYAlignment = 0xD378
wNorthConnectedMapXAlignment = 0xD379
wNorthConnectedMapViewPointer = 0xD37A
wMapConn2Ptr = 0xD37C
wSouthConnectionStripSrc = 0xD37D
wSouthConnectionStripDest = 0xD37F
wSouthConnectionStripWidth = 0xD381
wSouthConnectedMapWidth = 0xD382
wSouthConnectedMapYAlignment = 0xD383
wSouthConnectedMapXAlignment = 0xD384
wSouthConnectedMapViewPointer = 0xD385
wMapConn3Ptr = 0xD387
wWestConnectionStripSrc = 0xD388
wWestConnectionStripDest = 0xD38A
wWestConnectionStripHeight = 0xD38C
wWestConnectedMapWidth = 0xD38D
wWestConnectedMapYAlignment = 0xD38E
wWestConnectedMapXAlignment = 0xD38F
wWestConnectedMapViewPointer = 0xD390
wMapConn4Ptr = 0xD392
wEastConnectionStripSrc = 0xD393
wEastConnectionStripDest = 0xD395
wEastConnectionStripHeight = 0xD397
wEastConnectedMapWidth = 0xD398
wEastConnectedMapYAlignment = 0xD399
wEastConnectedMapXAlignment = 0xD39A
wEastConnectedMapViewPointer = 0xD39B
wSpriteSet = 0xD39D
wSpriteSetID = 0xD3A8
wObjectDataPointerTemp = 0xD3A9
wMapBackgroundTile = 0xD3AD
wNumberOfWarps = 0xD3AE
wWarpEntries = 0xD3AF
wDestinationWarpID = 0xD42F
wNumSigns = 0xD4B0
wSignCoords = 0xD4B1
wSignTextIDs = 0xD4D1
wNumSprites = 0xD4E1
wYOffsetSinceLastSpecialWarp = 0xD4E2
wXOffsetSinceLastSpecialWarp = 0xD4E3
wMapSpriteData = 0xD4E4
wMapSpriteExtraData = 0xD504
wCurrentMapHeight2 = 0xD524
wCurrentMapWidth2 = 0xD525
wMapViewVRAMPointer = 0xD526
wPlayerMovingDirection = 0xD528
wPlayerLastStopDirection = 0xD529
wPlayerDirection = 0xD52A
wTilesetBank = 0xD52B
wTilesetBlocksPtr = 0xD52C
wTilesetGfxPtr = 0xD52E
wTilesetCollisionPtr = 0xD530
wTilesetTalkingOverTiles = 0xD532
wGrassTile = 0xD535
wNumBoxItems = 0xD53A
wBoxItems = 0xD53B
wCurrentBoxNum = 0xD5A0
wNumHoFTeams = 0xD5A2
wUnusedD5A3 = 0xD5A3
wPlayerCoins = 0xD5A4
wMissableObjectFlags = 0xD5A6
wMissableObjectList = 0xD5CE
wGameProgressFlags = 0xD5F0
wOaksLabCurScript = 0xD5F0
wPalletTownCurScript = 0xD5F1
wBluesHouseCurScript = 0xD5F3
wViridianCityCurScript = 0xD5F4
wPewterCityCurScript = 0xD5F7
wRoute3CurScript = 0xD5F8
wRoute4CurScript = 0xD5F9
wViridianGymCurScript = 0xD5FB
wPewterGymCurScript = 0xD5FC
wCeruleanGymCurScript = 0xD5FD
wVermilionGymCurScript = 0xD5FE
wCeladonGymCurScript = 0xD5FF
wRoute6CurScript = 0xD600
wRoute8CurScript = 0xD601
wRoute24CurScript = 0xD602
wRoute25CurScript = 0xD603
wRoute9CurScript = 0xD604
wRoute10CurScript = 0xD605
wMtMoon1CurScript = 0xD606
wMtMoon3CurScript = 0xD607
wSSAnne8CurScript = 0xD608
wSSAnne9CurScript = 0xD609
wRoute22CurScript = 0xD60A
wRedsHouse2CurScript = 0xD60C
wViridianMarketCurScript = 0xD60D
wRoute22GateCurScript = 0xD60E
wCeruleanCityCurScript = 0xD60F
wSSAnne5CurScript = 0xD617
wViridianForestCurScript = 0xD618
wMuseum1fCurScript = 0xD619
wRoute13CurScript = 0xD61A
wRoute14CurScript = 0xD61B
wRoute17CurScript = 0xD61C
wRoute19CurScript = 0xD61D
wRoute21CurScript = 0xD61E
wSafariZoneEntranceCurScript = 0xD61F
wRockTunnel2CurScript = 0xD620
wRockTunnel1CurScript = 0xD621
wRoute11CurScript = 0xD623
wRoute12CurScript = 0xD624
wRoute15CurScript = 0xD625
wRoute16CurScript = 0xD626
wRoute18CurScript = 0xD627
wRoute20CurScript = 0xD628
wSSAnne10CurScript = 0xD629
wVermilionCityCurScript = 0xD62A
wPokemonTower2CurScript = 0xD62B
wPokemonTower3CurScript = 0xD62C
wPokemonTower4CurScript = 0xD62D
wPokemonTower5CurScript = 0xD62E
wPokemonTower6CurScript = 0xD62F
wPokemonTower7CurScript = 0xD630
wRocketHideout1CurScript = 0xD631
wRocketHideout2CurScript = 0xD632
wRocketHideout3CurScript = 0xD633
wRocketHideout4CurScript = 0xD634
wRoute6GateCurScript = 0xD636
wRoute8GateCurScript = 0xD637
wCinnabarIslandCurScript = 0xD639
wMansion1CurScript = 0xD63A
wMansion2CurScript = 0xD63C
wMansion3CurScript = 0xD63D
wMansion4CurScript = 0xD63E
wVictoryRoad2CurScript = 0xD63F
wVictoryRoad3CurScript = 0xD640
wFightingDojoCurScript = 0xD642
wSilphCo2CurScript = 0xD643
wSilphCo3CurScript = 0xD644
wSilphCo4CurScript = 0xD645
wSilphCo5CurScript = 0xD646
wSilphCo6CurScript = 0xD647
wSilphCo7CurScript = 0xD648
wSilphCo8CurScript = 0xD649
wSilphCo9CurScript = 0xD64A
wHallOfFameRoomCurScript = 0xD64B
wGaryCurScript = 0xD64C
wLoreleiCurScript = 0xD64D
wBrunoCurScript = 0xD64E
wAgathaCurScript = 0xD64F
wUnknownDungeon3CurScript = 0xD650
wVictoryRoad1CurScript = 0xD651
wLanceCurScript = 0xD653
wSilphCo10CurScript = 0xD658
wSilphCo11CurScript = 0xD659
wFuchsiaGymCurScript = 0xD65B
wSaffronGymCurScript = 0xD65C
wCinnabarGymCurScript = 0xD65E
wCeladonGameCornerCurScript = 0xD65F
wRoute16GateCurScript = 0xD660
wBillsHouseCurScript = 0xD661
wRoute5GateCurScript = 0xD662
wPowerPlantCurScript = 0xD663
wRoute7GateCurScript = 0xD663
wSSAnne2CurScript = 0xD665
wSeafoamIslands4CurScript = 0xD666
wRoute23CurScript = 0xD667
wSeafoamIslands5CurScript = 0xD668
wRoute18GateCurScript = 0xD669
wWalkBikeSurfState = 0xD700
wTownVisitedFlag = 0xD70B
wSafariSteps = 0xD70D
wFossilItem = 0xD70F
wFossilMon = 0xD710
wEnemyMonOrTrainerClass = 0xD713
wPlayerJumpingYScreenCoordsIndex = 0xD714
wRivalStarter = 0xD715
wPlayerStarter = 0xD717
wBoulderSpriteIndex = 0xD718
wLastBlackoutMap = 0xD719
wDestinationMap = 0xD71A
wUnusedD71B = 0xD71B
wTileInFrontOfBoulderAndBoulderCollisionResult = 0xD71C
wDungeonWarpDestinationMap = 0xD71D
wWhichDungeonWarp = 0xD71E
wUnusedD71F = 0xD71F
wd728 = 0xD728
wBeatGymFlags = 0xD72A
wd72c = 0xD72C
wd72d = 0xD72D
wd72e = 0xD72E
wd730 = 0xD730
wd732 = 0xD732
wFlags_D733 = 0xD733
wBeatLorelei = 0xD734
wd736 = 0xD736
wCompletedInGameTradeFlags = 0xD737
wWarpedFromWhichWarp = 0xD73B
wWarpedFromWhichMap = 0xD73C
wCardKeyDoorY = 0xD73F
wCardKeyDoorX = 0xD740
wFirstLockTrashCanIndex = 0xD743
wSecondLockTrashCanIndex = 0xD744
wEventFlags = 0xD747
wLinkEnemyTrainerName = 0xD887
wGrassRate = 0xD887
wGrassMons = 0xD888
wSerialEnemyDataBlock = 0xD893
wEnemyMons = 0xD8A4
wTrainerHeaderPtr = 0xDA30
wOpponentAfterWrongAnswer = 0xDA38
wUnusedDA38 = 0xDA38
wCurMapScript = 0xDA39
wPlayTimeHours = 0xDA41
wPlayTimeMaxed = 0xDA42
wPlayTimeMinutes = 0xDA43
wPlayTimeSeconds = 0xDA44
wPlayTimeFrames = 0xDA45
wSafariZoneGameOver = 0xDA46
wNumSafariBalls = 0xDA47
wDayCareInUse = 0xDA48
wBoxMonNicksEnd = 0xDEE2
wStack = 0xDFFF

pokemon_data = [
    {"hex": "1", "decimal": "1", "name": "Rhydon"},
    {"hex": "2", "decimal": "2", "name": "Kangaskhan"},
    {"hex": "3", "decimal": "3", "name": "Nidoran♂"},
    {"hex": "4", "decimal": "4", "name": "Clefairy"},
    {"hex": "5", "decimal": "5", "name": "Spearow"},
    {"hex": "6", "decimal": "6", "name": "Voltorb", "type": "Electric"},
    {"hex": "7", "decimal": "7", "name": "Nidoking"},
    {"hex": "8", "decimal": "8", "name": "Slowbro"},
    {"hex": "9", "decimal": "9", "name": "Ivysaur"},
    {"hex": "A", "decimal": "10", "name": "Exeggutor"},
    {"hex": "B", "decimal": "11", "name": "Lickitung"},
    {"hex": "C", "decimal": "12", "name": "Exeggcute"},
    {"hex": "D", "decimal": "13", "name": "Grimer"},
    {"hex": "E", "decimal": "14", "name": "Gengar", "type": "Ghost"},
    {"hex": "F", "decimal": "15", "name": "Nidoran♀"},
    {"hex": "10", "decimal": "16", "name": "Nidoqueen"},
    {"hex": "11", "decimal": "17", "name": "Cubone"},
    {"hex": "12", "decimal": "18", "name": "Rhyhorn"},
    {"hex": "13", "decimal": "19", "name": "Lapras", "type": "Ice"},
    {"hex": "14", "decimal": "20", "name": "Arcanine"},
    {"hex": "15", "decimal": "21", "name": "Mew"},
    {"hex": "16", "decimal": "22", "name": "Gyarados"},
    {"hex": "17", "decimal": "23", "name": "Shellder"},
    {"hex": "18", "decimal": "24", "name": "Tentacool"},
    {"hex": "19", "decimal": "25", "name": "Gastly", "type": "Ghost"},
    {"hex": "1A", "decimal": "26", "name": "Scyther", "type": "Bug"},
    {"hex": "1B", "decimal": "27", "name": "Staryu"},
    {"hex": "1C", "decimal": "28", "name": "Blastoise"},
    {"hex": "1D", "decimal": "29", "name": "Pinsir", "type": "Bug"},
    {"hex": "1E", "decimal": "30", "name": "Tangela"},
    {"hex": "1F", "decimal": "31", "name": "MissingNo. (Scizor)"},
    {"hex": "20", "decimal": "32", "name": "MissingNo. (Shuckle)"},
    {"hex": "21", "decimal": "33", "name": "Growlithe"},
    {"hex": "22", "decimal": "34", "name": "Onix"},
    {"hex": "23", "decimal": "35", "name": "Fearow"},
    {"hex": "24", "decimal": "36", "name": "Pidgey"},
    {"hex": "25", "decimal": "37", "name": "Slowpoke"},
    {"hex": "26", "decimal": "38", "name": "Kadabra"},
    {"hex": "27", "decimal": "39", "name": "Graveler"},
    {"hex": "28", "decimal": "40", "name": "Chansey"},
    {"hex": "29", "decimal": "41", "name": "Machoke"},
    {"hex": "2A", "decimal": "42", "name": "Mr. Mime"},
    {"hex": "2B", "decimal": "43", "name": "Hitmonlee"},
    {"hex": "2C", "decimal": "44", "name": "Hitmonchan"},
    {"hex": "2D", "decimal": "45", "name": "Arbok"},
    {"hex": "2E", "decimal": "46", "name": "Parasect", "type": "Bug"},
    {"hex": "2F", "decimal": "47", "name": "Psyduck"},
    {"hex": "30", "decimal": "48", "name": "Drowzee"},
    {"hex": "31", "decimal": "49", "name": "Golem"},
    {"hex": "32", "decimal": "50", "name": "MissingNo. (Heracross)"},
    {"hex": "33", "decimal": "51", "name": "Magmar"},
    {"hex": "34", "decimal": "52", "name": "MissingNo. (Ho-Oh)"},
    {"hex": "35", "decimal": "53", "name": "Electabuzz", "type": "Electric"},
    {"hex": "36", "decimal": "54", "name": "Magneton", "type": "Electric"},
    {"hex": "37", "decimal": "55", "name": "Koffing"},
    {"hex": "38", "decimal": "56", "name": "MissingNo. (Sneasel)"},
    {"hex": "39", "decimal": "57", "name": "Mankey"},
    {"hex": "3A", "decimal": "58", "name": "Seel"},
    {"hex": "3B", "decimal": "59", "name": "Diglett"},
    {"hex": "3C", "decimal": "60", "name": "Tauros"},
    {"hex": "3D", "decimal": "61", "name": "MissingNo. (Teddiursa)"},
    {"hex": "3E", "decimal": "62", "name": "MissingNo. (Ursaring)"},
    {"hex": "3F", "decimal": "63", "name": "MissingNo. (Slugma)"},
    {"hex": "40", "decimal": "64", "name": "Farfetch'd"},
    {"hex": "41", "decimal": "65", "name": "Venonat", "type": "Bug"},
    {"hex": "42", "decimal": "66", "name": "Dragonite", "type": "Dragon"},
    {"hex": "43", "decimal": "67", "name": "MissingNo. (Magcargo)"},
    {"hex": "44", "decimal": "68", "name": "MissingNo. (Swinub)"},
    {"hex": "45", "decimal": "69", "name": "MissingNo. (Piloswine)"},
    {"hex": "46", "decimal": "70", "name": "Doduo"},
    {"hex": "47", "decimal": "71", "name": "Poliwag"},
    {"hex": "48", "decimal": "72", "name": "Jynx", "type": "Ice"},
    {"hex": "49", "decimal": "73", "name": "Moltres"},
    {"hex": "4A", "decimal": "74", "name": "Articuno", "type": "Ice"},
    {"hex": "4B", "decimal": "75", "name": "Zapdos", "type": "Electric"},
    {"hex": "4C", "decimal": "76", "name": "Ditto"},
    {"hex": "4D", "decimal": "77", "name": "Meowth"},
    {"hex": "4E", "decimal": "78", "name": "Krabby"},
    {"hex": "4F", "decimal": "79", "name": "MissingNo. (Corsola)"},
    {"hex": "50", "decimal": "80", "name": "MissingNo. (Remoraid)"},
    {"hex": "51", "decimal": "81", "name": "MissingNo. (Octillery)"},
    {"hex": "52", "decimal": "82", "name": "Vulpix"},
    {"hex": "53", "decimal": "83", "name": "Ninetales"},
    {"hex": "54", "decimal": "84", "name": "Pikachu", "type": "Electric"},
    {"hex": "55", "decimal": "85", "name": "Raichu", "type": "Electric"},
    {"hex": "56", "decimal": "86", "name": "MissingNo. (Deli)"},
    {"hex": "57", "decimal": "87", "name": "MissingNo. (Mantine)"},
    {"hex": "58", "decimal": "88", "name": "Dratini", "type": "Dragon"},
    {"hex": "59", "decimal": "89", "name": "Dragonair", "type": "Dragon"},
    {"hex": "5A", "decimal": "90", "name": "Kabuto"},
    {"hex": "5B", "decimal": "91", "name": "Kabutops"},
    {"hex": "5C", "decimal": "92", "name": "Horsea"},
    {"hex": "5D", "decimal": "93", "name": "Seadra"},
    {"hex": "5E", "decimal": "94", "name": "MissingNo. (Skarmory)"},
    {"hex": "5F", "decimal": "95", "name": "MissingNo. (Houndour)"},
    {"hex": "60", "decimal": "96", "name": "Sandshrew"},
    {"hex": "61", "decimal": "97", "name": "Sandslash"},
    {"hex": "62", "decimal": "98", "name": "Omanyte"},
    {"hex": "63", "decimal": "99", "name": "Omastar"},
    {"hex": "64", "decimal": "100", "name": "Jigglypuff"},
    {"hex": "65", "decimal": "101", "name": "Wigglytuff"},
    {"hex": "66", "decimal": "102", "name": "Eevee"},
    {"hex": "67", "decimal": "103", "name": "Flareon"},
    {"hex": "68", "decimal": "104", "name": "Jolteon", "type": "Electric"},
    {"hex": "69", "decimal": "105", "name": "Vaporeon"},
    {"hex": "6A", "decimal": "106", "name": "Machop"},
    {"hex": "6B", "decimal": "107", "name": "Zubat"},
    {"hex": "6C", "decimal": "108", "name": "Ekans"},
    {"hex": "6D", "decimal": "109", "name": "Paras", "type": "Bug"},
    {"hex": "6E", "decimal": "110", "name": "Poliwhirl"},
    {"hex": "6F", "decimal": "111", "name": "Poliwrath"},
    {"hex": "70", "decimal": "112", "name": "Weedle", "type": "Bug"},
    {"hex": "71", "decimal": "113", "name": "Kakuna", "type": "Bug"},
    {"hex": "72", "decimal": "114", "name": "Beedrill", "type": "Bug"},
    {"hex": "73", "decimal": "115", "name": "MissingNo. (Houndoom)"},
    {"hex": "74", "decimal": "116", "name": "Dodrio"},
    {"hex": "75", "decimal": "117", "name": "Primeape"},
    {"hex": "76", "decimal": "118", "name": "Dugtrio"},
    {"hex": "77", "decimal": "119", "name": "Venomoth", "type": "Bug"},
    {"hex": "78", "decimal": "120", "name": "Dewgong", "type": "Ice"},
    {"hex": "79", "decimal": "121", "name": "MissingNo. (Kingdra)"},
    {"hex": "7A", "decimal": "122", "name": "MissingNo. (Phanpy)"},
    {"hex": "7B", "decimal": "123", "name": "Caterpie", "type": "Bug"},
    {"hex": "7C", "decimal": "124", "name": "Metapod", "type": "Bug"},
    {"hex": "7D", "decimal": "125", "name": "Butterfree", "type": "Bug"},
    {"hex": "7E", "decimal": "126", "name": "Machamp"},
    {"hex": "7F", "decimal": "127", "name": "MissingNo. (Donphan)"},
    {"hex": "80", "decimal": "128", "name": "Golduck"},
    {"hex": "81", "decimal": "129", "name": "Hypno"},
    {"hex": "82", "decimal": "130", "name": "Golbat"},
    {"hex": "83", "decimal": "131", "name": "Mewtwo"},
    {"hex": "84", "decimal": "132", "name": "Snorlax"},
    {"hex": "85", "decimal": "133", "name": "Magikarp"},
    {"hex": "86", "decimal": "134", "name": "MissingNo. (Porygon2)"},
    {"hex": "87", "decimal": "135", "name": "MissingNo. (Stantler)"},
    {"hex": "88", "decimal": "136", "name": "Muk"},
    {"hex": "89", "decimal": "137", "name": "MissingNo. (Smeargle)"},
    {"hex": "8A", "decimal": "138", "name": "Kingler"},
    {"hex": "8B", "decimal": "139", "name": "Cloyster"},
    {"hex": "8D", "decimal": "141", "name": "Electrode"},
    {"hex": "8E", "decimal": "142", "name": "Clefable"},
    {"hex": "8F", "decimal": "143", "name": "Weezing"},
    {"hex": "90", "decimal": "144", "name": "Persian"},
    {"hex": "91", "decimal": "145", "name": "Marowak"},
    {"hex": "93", "decimal": "147", "name": "Haunter"},
    {"hex": "94", "decimal": "148", "name": "Abra"},
    {"hex": "95", "decimal": "149", "name": "Alakazam"},
    {"hex": "96", "decimal": "150", "name": "Pidgeotto"},
    {"hex": "97", "decimal": "151", "name": "Pidgeot"},
    {"hex": "98", "decimal": "152", "name": "Starmie"},
    {"hex": "99", "decimal": "153", "name": "Bulbasaur"},
    {"hex": "9A", "decimal": "154", "name": "Venusaur"},
    {"hex": "9B", "decimal": "155", "name": "Tentacruel"},
    {"hex": "9D", "decimal": "157", "name": "Goldeen"},
    {"hex": "9E", "decimal": "158", "name": "Seaking"},
    {"hex": "A3", "decimal": "163", "name": "Ponyta"},
    {"hex": "A4", "decimal": "164", "name": "Rapidash"},
    {"hex": "A5", "decimal": "165", "name": "Rattata"},
    {"hex": "A6", "decimal": "166", "name": "Raticate"},
    {"hex": "A7", "decimal": "167", "name": "Nidorino"},
    {"hex": "A8", "decimal": "168", "name": "Nidorina"},
    {"hex": "A9", "decimal": "169", "name": "Geodude"},
    {"hex": "AA", "decimal": "170", "name": "Porygon"},
    {"hex": "AB", "decimal": "171", "name": "Aerodactyl"},
    {"hex": "AD", "decimal": "173", "name": "Magnemite"},
    {"hex": "B0", "decimal": "176", "name": "Charmander"},
    {"hex": "B1", "decimal": "177", "name": "Squirtle"},
    {"hex": "B2", "decimal": "178", "name": "Charmeleon"},
    {"hex": "B3", "decimal": "179", "name": "Wartortle"},
    {"hex": "B4", "decimal": "180", "name": "Charizard"},
    {"hex": "B9", "decimal": "185", "name": "Oddish"},
    {"hex": "BA", "decimal": "186", "name": "Gloom"},
    {"hex": "BB", "decimal": "187", "name": "Vileplume"},
    {"hex": "BC", "decimal": "188", "name": "Bellsprout"},
    {"hex": "BD", "decimal": "189", "name": "Weepinbell"},
    {"hex": "BE", "decimal": "190", "name": "Victreebel"},
]

MOVES_DICT = {
    1: {
        "Move": "Pound",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 40,
        "Acc": "100%",
    },
    2: {
        "Move": "Karate Chop",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 25,
        "Power": 50,
        "Acc": "100%",
    },
    3: {
        "Move": "Double Slap",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 10,
        "Power": 15,
        "Acc": "85%",
    },
    4: {
        "Move": "Comet Punch",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 18,
        "Acc": "85%",
    },
    5: {
        "Move": "Mega Punch",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 80,
        "Acc": "85%",
    },
    6: {
        "Move": "Pay Day",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 40,
        "Acc": "100%",
    },
    7: {
        "Move": "Fire Punch",
        "Type": "Fire",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 75,
        "Acc": "100%",
    },
    8: {
        "Move": "Ice Punch",
        "Type": "Ice",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 75,
        "Acc": "100%",
    },
    9: {
        "Move": "Thunder Punch",
        "Type": "Electric",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 75,
        "Acc": "100%",
    },
    10: {
        "Move": "Scratch",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 40,
        "Acc": "100%",
    },
    11: {
        "Move": "Vise Grip",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 30,
        "Power": 55,
        "Acc": "100%",
    },
    12: {
        "Move": "Guillotine",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 5,
        "Power": "—",
        "Acc": "30%",
    },
    13: {
        "Move": "Razor Wind",
        "Type": "Normal",
        "Phy/Spec": "Special",
        "PP": 10,
        "Power": 80,
        "Acc": "100%",
    },
    14: {
        "Move": "Swords Dance",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 20,
        "Power": "—",
        "Acc": "—%",
    },
    15: {
        "Move": "Cut",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 30,
        "Power": 50,
        "Acc": "95%",
    },
    16: {
        "Move": "Gust",
        "Type": "Flying",
        "Phy/Spec": "Special",
        "PP": 35,
        "Power": 40,
        "Acc": "100%",
    },
    17: {
        "Move": "Wing Attack",
        "Type": "Flying",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 60,
        "Acc": "100%",
    },
    18: {
        "Move": "Whirlwind",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 20,
        "Power": "—",
        "Acc": "—%",
    },
    19: {
        "Move": "Fly",
        "Type": "Flying",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 90,
        "Acc": "95%",
    },
    20: {
        "Move": "Bind",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 15,
        "Acc": "85%",
    },
    21: {
        "Move": "Slam",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 80,
        "Acc": "75%",
    },
    22: {
        "Move": "Vine Whip",
        "Type": "Grass",
        "Phy/Spec": "Physical",
        "PP": 25,
        "Power": 45,
        "Acc": "100%",
    },
    23: {
        "Move": "Stomp",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 65,
        "Acc": "100%",
    },
    24: {
        "Move": "Double Kick",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 30,
        "Power": 30,
        "Acc": "100%",
    },
    25: {
        "Move": "Mega Kick",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 5,
        "Power": 120,
        "Acc": "75%",
    },
    26: {
        "Move": "Jump Kick",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 10,
        "Power": 100,
        "Acc": "95%",
    },
    27: {
        "Move": "Rolling Kick",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 60,
        "Acc": "85%",
    },
    28: {
        "Move": "Sand Attack",
        "Type": "Ground",
        "Phy/Spec": "Status",
        "PP": 15,
        "Power": "—",
        "Acc": "100%",
    },
    29: {
        "Move": "Headbutt",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 70,
        "Acc": "100%",
    },
    30: {
        "Move": "Horn Attack",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 25,
        "Power": 65,
        "Acc": "100%",
    },
    31: {
        "Move": "Fury Attack",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 15,
        "Acc": "85%",
    },
    32: {
        "Move": "Horn Drill",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 5,
        "Power": "—",
        "Acc": "30%",
    },
    33: {
        "Move": "Tackle",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 40,
        "Acc": "100%",
    },
    34: {
        "Move": "Body Slam",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 85,
        "Acc": "100%",
    },
    35: {
        "Move": "Wrap",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 15,
        "Acc": "90%",
    },
    36: {
        "Move": "Take Down",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 90,
        "Acc": "85%",
    },
    37: {
        "Move": "Thrash",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 10,
        "Power": 120,
        "Acc": "100%",
    },
    38: {
        "Move": "Double-Edge",
        "Type": "Normal",
        "Phy/Spec": "Physical",
        "PP": 15,
        "Power": 120,
        "Acc": "100%",
    },
    39: {
        "Move": "Tail Whip",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 30,
        "Power": "—",
        "Acc": "100%",
    },
    40: {
        "Move": "Poison Sting",
        "Type": "Poison",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 15,
        "Acc": "100%",
    },
    41: {
        "Move": "Twineedle",
        "Type": "Bug",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 25,
        "Acc": "100%",
    },
    42: {
        "Move": "Pin Missile",
        "Type": "Bug",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 25,
        "Acc": "95%",
    },
    43: {
        "Move": "Leer",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 30,
        "Power": "—",
        "Acc": "100%",
    },
    44: {
        "Move": "Bite",
        "Type": "Dark",
        "Phy/Spec": "Physical",
        "PP": 25,
        "Power": 60,
        "Acc": "100%",
    },
    45: {
        "Move": "Growl",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 40,
        "Power": "—",
        "Acc": "100%",
    },
    46: {
        "Move": "Roar",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 20,
        "Power": "—",
        "Acc": "—%",
    },
    47: {
        "Move": "Sing",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 15,
        "Power": "—",
        "Acc": "55%",
    },
    48: {
        "Move": "Supersonic",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 20,
        "Power": "—",
        "Acc": "55%",
    },
    49: {
        "Move": "Sonic Boom",
        "Type": "Normal",
        "Phy/Spec": "Special",
        "PP": 20,
        "Power": "—",
        "Acc": "90%",
    },
    50: {
        "Move": "Disable",
        "Type": "Normal",
        "Phy/Spec": "Status",
        "PP": 20,
        "Power": "—",
        "Acc": "100%",
    },
    51: {
        "Move": "Acid",
        "Type": "Poison",
        "Phy/Spec": "Special",
        "PP": 30,
        "Power": 40,
        "Acc": "100%",
    },
    52: {
        "Move": "Ember",
        "Type": "Fire",
        "Phy/Spec": "Special",
        "PP": 25,
        "Power": 40,
        "Acc": "100%",
    },
    53: {
        "Move": "Flamethrower",
        "Type": "Fire",
        "Phy/Spec": "Special",
        "PP": 15,
        "Power": 90,
        "Acc": "100%",
    },
    54: {
        "Move": "Mist",
        "Type": "Ice",
        "Phy/Spec": "Status",
        "PP": 30,
        "Power": "—",
        "Acc": "—%",
    },
    55: {
        "Move": "Water Gun",
        "Type": "Water",
        "Phy/Spec": "Special",
        "PP": 25,
        "Power": 40,
        "Acc": "100%",
    },
    56: {
        "Move": "Hydro Pump",
        "Type": "Water",
        "Phy/Spec": "Special",
        "PP": 5,
        "Power": 110,
        "Acc": "80%",
    },
    57: {
        "Move": "Surf",
        "Type": "Water",
        "Phy/Spec": "Special",
        "PP": 15,
        "Power": 90,
        "Acc": "100%",
    },
    58: {
        "Move": "Ice Beam",
        "Type": "Ice",
        "Phy/Spec": "Special",
        "PP": 10,
        "Power": 90,
        "Acc": "100%",
    },
    59: {
        "Move": "Blizzard",
        "Type": "Ice",
        "Phy/Spec": "Special",
        "PP": 5,
        "Power": 110,
        "Acc": "70%",
    },
    60: {
        "Move": "Psybeam",
        "Type": "Psychic",
        "Phy/Spec": "Special",
        "PP": 20,
        "Power": 65,
        "Acc": "100%",
    },
    61: {
        "Move": "Bubble Beam",
        "Type": "Water",
        "Phy/Spec": "Special",
        "PP": 20,
        "Power": 65,
        "Acc": "100%",
    },
    62: {
        "Move": "Aurora Beam",
        "Type": "Ice",
        "Phy/Spec": "Special",
        "PP": 20,
        "Power": 65,
        "Acc": "100%",
    },
    63: {
        "Move": "Hyper Beam",
        "Type": "Normal",
        "Phy/Spec": "Special",
        "PP": 5,
        "Power": 150,
        "Acc": "90%",
    },
    64: {
        "Move": "Peck",
        "Type": "Flying",
        "Phy/Spec": "Physical",
        "PP": 35,
        "Power": 35,
        "Acc": "100%",
    },
    65: {
        "Move": "Drill Peck",
        "Type": "Flying",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 80,
        "Acc": "100%",
    },
    66: {
        "Move": "Submission",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": 80,
        "Acc": "80%",
    },
    67: {
        "Move": "Low Kick",
        "Type": "Fighting",
        "Phy/Spec": "Physical",
        "PP": 20,
        "Power": "—",
        "Acc": "100%",
    },
    68: {
        "Move": "Counter",
        "Type": "Fighting",
        "Category": "Physical",
        "PP": 20,
        "Power": "—",
        "Accuracy": "100%",
    },
    69: {
        "Move": "Seismic Toss",
        "Type": "Fighting",
        "Category": "Physical",
        "PP": 20,
        "Power": "—",
        "Accuracy": "100%",
    },
    70: {
        "Move": "Strength",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 15,
        "Power": 80,
        "Accuracy": "100%",
    },
    71: {
        "Move": "Absorb",
        "Type": "Grass",
        "Category": "Special",
        "PP": 25,
        "Power": 20,
        "Accuracy": "100%",
    },
    72: {
        "Move": "Mega Drain",
        "Type": "Grass",
        "Category": "Special",
        "PP": 15,
        "Power": 40,
        "Accuracy": "100%",
    },
    73: {
        "Move": "Leech Seed",
        "Type": "Grass",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "90%",
    },
    74: {
        "Move": "Growth",
        "Type": "Normal",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    75: {
        "Move": "Razor Leaf",
        "Type": "Grass",
        "Category": "Physical",
        "PP": 25,
        "Power": 55,
        "Accuracy": "95%",
    },
    76: {
        "Move": "Solar Beam",
        "Type": "Grass",
        "Category": "Special",
        "PP": 10,
        "Power": 120,
        "Accuracy": "100%",
    },
    77: {
        "Move": "Poison Powder",
        "Type": "Poison",
        "Category": "Status",
        "PP": 35,
        "Power": "—",
        "Accuracy": "75%",
    },
    78: {
        "Move": "Stun Spore",
        "Type": "Grass",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "75%",
    },
    79: {
        "Move": "Sleep Powder",
        "Type": "Grass",
        "Category": "Status",
        "PP": 15,
        "Power": "—",
        "Accuracy": "75%",
    },
    80: {
        "Move": "Petal Dance",
        "Type": "Grass",
        "Category": "Special",
        "PP": 10,
        "Power": 120,
        "Accuracy": "100%",
    },
    81: {
        "Move": "String Shot",
        "Type": "Bug",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": "95%",
    },
    82: {
        "Move": "Dragon Rage",
        "Type": "Dragon",
        "Category": "Special",
        "PP": 10,
        "Power": "—",
        "Accuracy": "100%",
    },
    83: {
        "Move": "Fire Spin",
        "Type": "Fire",
        "Category": "Special",
        "PP": 15,
        "Power": 35,
        "Accuracy": "85%",
    },
    84: {
        "Move": "Thunder Shock",
        "Type": "Electric",
        "Category": "Special",
        "PP": 30,
        "Power": 40,
        "Accuracy": "100%",
    },
    85: {
        "Move": "Thunderbolt",
        "Type": "Electric",
        "Category": "Special",
        "PP": 15,
        "Power": 90,
        "Accuracy": "100%",
    },
    86: {
        "Move": "Thunder Wave",
        "Type": "Electric",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "90%",
    },
    87: {
        "Move": "Thunder",
        "Type": "Electric",
        "Category": "Special",
        "PP": 10,
        "Power": 110,
        "Accuracy": "70%",
    },
    88: {
        "Move": "Rock Throw",
        "Type": "Rock",
        "Category": "Physical",
        "PP": 15,
        "Power": 50,
        "Accuracy": "90%",
    },
    89: {
        "Move": "Earthquake",
        "Type": "Ground",
        "Category": "Physical",
        "PP": 10,
        "Power": 100,
        "Accuracy": "100%",
    },
    90: {
        "Move": "Fissure",
        "Type": "Ground",
        "Category": "Physical",
        "PP": 5,
        "Power": "—",
        "Accuracy": "30%",
    },
    91: {
        "Move": "Dig",
        "Type": "Ground",
        "Category": "Physical",
        "PP": 10,
        "Power": 80,
        "Accuracy": "100%",
    },
    92: {
        "Move": "Toxic",
        "Type": "Poison",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "90%",
    },
    93: {
        "Move": "Confusion",
        "Type": "Psychic",
        "Category": "Special",
        "PP": 25,
        "Power": 50,
        "Accuracy": "100%",
    },
    94: {
        "Move": "Psychic",
        "Type": "Psychic",
        "Category": "Special",
        "PP": 10,
        "Power": 90,
        "Accuracy": "100%",
    },
    95: {
        "Move": "Hypnosis",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "60%",
    },
    96: {
        "Move": "Meditate",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": "—%",
    },
    97: {
        "Move": "Agility",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "—%",
    },
    98: {
        "Move": "Quick Attack",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 30,
        "Power": 40,
        "Accuracy": "100%",
    },
    99: {
        "Move": "Rage",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 20,
        "Power": 20,
        "Accuracy": "100%",
    },
    100: {
        "Move": "Teleport",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    101: {
        "Move": "Night Shade",
        "Type": "Ghost",
        "Category": "Special",
        "PP": 15,
        "Power": "—",
        "Accuracy": "100%",
    },
    102: {
        "Move": "Mimic",
        "Type": "Normal",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "—%",
    },
    103: {
        "Move": "Screech",
        "Type": "Normal",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": "85%",
    },
    104: {
        "Move": "Double Team",
        "Type": "Normal",
        "Category": "Status",
        "PP": 15,
        "Power": "—",
        "Accuracy": "—%",
    },
    105: {
        "Move": "Recover",
        "Type": "Normal",
        "Category": "Status",
        "PP": 5,
        "Power": "—",
        "Accuracy": "—%",
    },
    106: {
        "Move": "Harden",
        "Type": "Normal",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "—%",
    },
    107: {
        "Move": "Minimize",
        "Type": "Normal",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "—%",
    },
    108: {
        "Move": "Smokescreen",
        "Type": "Normal",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "100%",
    },
    109: {
        "Move": "Confuse Ray",
        "Type": "Ghost",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "100%",
    },
    110: {
        "Move": "Withdraw",
        "Type": "Water",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": "—%",
    },
    111: {
        "Move": "Defense Curl",
        "Type": "Normal",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": "—%",
    },
    112: {
        "Move": "Barrier",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    113: {
        "Move": "Light Screen",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "—%",
    },
    114: {
        "Move": "Haze",
        "Type": "Ice",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "—%",
    },
    115: {
        "Move": "Reflect",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    116: {
        "Move": "Focus Energy",
        "Type": "Normal",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "—%",
    },
    117: {
        "Move": "Bide",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 10,
        "Power": "—",
        "Accuracy": "—%",
    },
    118: {
        "Move": "Metronome",
        "Type": "Normal",
        "Category": "Status",
        "PP": 10,
        "Power": "—",
        "Accuracy": "—%",
    },
    119: {
        "Move": "Mirror Move",
        "Type": "Flying",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    120: {
        "Move": "Self-Destruct",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 5,
        "Power": 200,
        "Accuracy": "100%",
    },
    121: {
        "Move": "Egg Bomb",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 10,
        "Power": 100,
        "Accuracy": "75%",
    },
    122: {
        "Move": "Lick",
        "Type": "Ghost",
        "Category": "Physical",
        "PP": 30,
        "Power": 30,
        "Accuracy": "100%",
    },
    123: {
        "Move": "Smog",
        "Type": "Poison",
        "Category": "Special",
        "PP": 20,
        "Power": 30,
        "Accuracy": "70%",
    },
    124: {
        "Move": "Sludge",
        "Type": "Poison",
        "Category": "Special",
        "PP": 20,
        "Power": 65,
        "Accuracy": "100%",
    },
    125: {
        "Move": "Bone Club",
        "Type": "Ground",
        "Category": "Physical",
        "PP": 20,
        "Power": 65,
        "Accuracy": "85%",
    },
    126: {
        "Move": "Fire Blast",
        "Type": "Fire",
        "Category": "Special",
        "PP": 5,
        "Power": 110,
        "Accuracy": "85%",
    },
    127: {
        "Move": "Waterfall",
        "Type": "Water",
        "Category": "Physical",
        "PP": 15,
        "Power": 80,
        "Accuracy": "100%",
    },
    128: {
        "Move": "Clamp",
        "Type": "Water",
        "Category": "Physical",
        "PP": 10,
        "Power": 35,
        "Accuracy": "85%",
    },
    129: {
        "Move": "Swift",
        "Type": "Normal",
        "Category": "Special",
        "PP": 20,
        "Power": 60,
        "Accuracy": "—%",
    },
    130: {
        "Move": "Skull Bash",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 10,
        "Power": 130,
        "Accuracy": "100%",
    },
    131: {
        "Move": "Spike Cannon",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 15,
        "Power": 20,
        "Accuracy": "100%",
    },
    132: {
        "Move": "Constrict",
        "Type": "Normal",
        "Category": "Physical",
        "PP": 35,
        "Power": 10,
        "Accuracy": "100%",
    },
    133: {
        "Move": "Amnesia",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 20,
        "Power": "—",
        "Accuracy": "—%",
    },
    134: {
        "Move": "Kinesis",
        "Type": "Psychic",
        "Category": "Status",
        "PP": 15,
        "Power": "—",
        "Accuracy": "80%",
    },
    135: {
        "Move": "Soft-Boiled",
        "Type": "Normal",
        "Category": "Status",
        "PP": 5,
        "Power": "—",
        "Accuracy": "—%",
    },
    136: {
        "Move": "High Jump Kick",
        "Type": "Fighting",
        "Category": "Physical",
        "PP": 10,
        "Power": 130,
        "Accuracy": "90%",
    },
    137: {
        "Move": "Glare",
        "Type": "Normal",
        "Category": "Status",
        "PP": 30,
        "Power": "—",
        "Accuracy": "100%",
    },
    138: {
        "Move": "Dream Eater",
        "Type": "Psychic",
        "Category": "Special",
        "PP": 15,
        "Power": 100,
        "Accuracy": "100%",
    },
    139: {
        "Move": "Poison Gas",
        "Type": "Poison",
        "Category": "Status",
        "PP": 40,
        "Power": "—",
        "Accuracy": 90,
    },
    140: {
        "Move": "Barrage",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 20,
        "PP": 15,
        "Accuracy": 85,
    },
    141: {
        "Move": "Leech Life",
        "Type": "Bug",
        "Category": "Physical",
        "Power": 10,
        "PP": 80,
        "Accuracy": 100,
    },
    142: {
        "Move": "Lovely Kiss",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 10,
        "Accuracy": 75,
    },
    143: {
        "Move": "Sky Attack",
        "Type": "Flying",
        "Category": "Physical",
        "Power": 140,
        "PP": 5,
        "Accuracy": 90,
    },
    144: {
        "Move": "Transform",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 10,
        "Accuracy": "—%",
    },
    145: {
        "Move": "Bubble",
        "Type": "Water",
        "Category": "Special",
        "Power": 40,
        "PP": 30,
        "Accuracy": 100,
    },
    146: {
        "Move": "Dizzy Punch",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 70,
        "PP": 10,
        "Accuracy": 100,
    },
    147: {
        "Move": "Spore",
        "Type": "Grass",
        "Category": "Status",
        "Power": "—",
        "PP": 15,
        "Accuracy": 100,
    },
    148: {
        "Move": "Flash",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 20,
        "Accuracy": 100,
    },
    149: {
        "Move": "Psywave",
        "Type": "Psychic",
        "Category": "Special",
        "Power": "—",
        "PP": 15,
        "Accuracy": 100,
    },
    150: {
        "Move": "Splash",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 40,
        "Accuracy": "—%",
    },
    151: {
        "Move": "Acid Armor",
        "Type": "Poison",
        "Category": "Status",
        "Power": "—",
        "PP": 20,
        "Accuracy": "—%",
    },
    152: {
        "Move": "Crabhammer",
        "Type": "Water",
        "Category": "Physical",
        "Power": 100,
        "PP": 10,
        "Accuracy": 90,
    },
    153: {
        "Move": "Explosion",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 250,
        "PP": 5,
        "Accuracy": 100,
    },
    154: {
        "Move": "Fury Swipes",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 18,
        "PP": 15,
        "Accuracy": 80,
    },
    155: {
        "Move": "Bonemerang",
        "Type": "Ground",
        "Category": "Physical",
        "Power": 50,
        "PP": 10,
        "Accuracy": 90,
    },
    156: {
        "Move": "Rest",
        "Type": "Psychic",
        "Category": "Status",
        "Power": "—",
        "PP": 10,
        "Accuracy": "—%",
    },
    157: {
        "Move": "Rock Slide",
        "Type": "Rock",
        "Category": "Physical",
        "Power": 75,
        "PP": 10,
        "Accuracy": 90,
    },
    158: {
        "Move": "Hyper Fang",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 80,
        "PP": 15,
        "Accuracy": 90,
    },
    159: {
        "Move": "Sharpen",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 30,
        "Accuracy": "—%",
    },
    160: {
        "Move": "Conversion",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 30,
        "Accuracy": "—%",
    },
    161: {
        "Move": "Tri Attack",
        "Type": "Normal",
        "Category": "Special",
        "Power": 80,
        "PP": 10,
        "Accuracy": 100,
    },
    162: {
        "Move": "Super Fang",
        "Type": "Normal",
        "Category": "Physical",
        "Power": "—",
        "PP": 10,
        "Accuracy": 90,
    },
    163: {
        "Move": "Slash",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 70,
        "PP": 20,
        "Accuracy": 100,
    },
    164: {
        "Move": "Substitute",
        "Type": "Normal",
        "Category": "Status",
        "Power": "—",
        "PP": 10,
        "Accuracy": "—%",
    },
    165: {
        "Move": "Struggle",
        "Type": "Normal",
        "Category": "Physical",
        "Power": 50,
        "PP": "—",
        "Accuracy": "—%",
    },
}


# def read_m(pyboy, addr: str | int) -> int:
#     if isinstance(addr, str):
#         _, addr = symbol_lookup(pyboy, addr)
#     return pyboy.memory[addr]


def read_m(pyboy, addr: str | int) -> int:
    if isinstance(addr, str):
        return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
    return pyboy.memory[addr]


def memory(pyboy, addr):
    return pyboy.memory[addr]


def read_short(pyboy, addr: str | int) -> int:
    if isinstance(addr, str):
        _, addr = symbol_lookup(pyboy, addr)
    data = pyboy.memory[addr : addr + 2]
    return (data[0] << 8) + data[1]


def bcd(num):
    return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)


def bit_count(bits):
    return bin(bits).count("1")


# def read_bit(pyboy, addr, bit) -> bool:
#     # add padding so zero will read '0b100000000' instead of '0b0'
#     return bin(256 + read_m(pyboy, addr))[-bit - 1] == "1"


def read_uint16(pyboy, start_addr):
    """Read 2 bytes"""
    val_256 = read_m(pyboy, start_addr)
    val_1 = read_m(pyboy, start_addr + 1)
    return 256 * val_256 + val_1


STATUSDICT = {
    0x08: "Poison",
    # 0x04: 'Burn',
    # 0x05: 'Frozen',
    # 0x06: 'Paralyze',
    0x00: "None",
}
POKE = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]  # - Pokémon (Again)
STATUS = [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]  # - Status (Poisoned, Paralyzed, etc.)
TYPE1 = [0xD170, 0xD19C, 0xD1C8, 0xD1F4, 0xD220, 0xD24C]  # - Type 1
TYPE2 = [0xD171, 0xD19D, 0xD1C9, 0xD1F5, 0xD221, 0xD24D]  # - Type 2
LEVEL = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]  # - Level (actual level)
MAXHP = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]  # - Max HP if = 01 + 256 to MAXHP2 value
CHP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]  # - Current HP if = 01 + 256


def pokemon(pyboy):
    # Get memory values from the list POKE and LEVEL
    memory_values = [read_m(pyboy, a) for a in POKE]
    levels = [read_m(pyboy, a) for a in LEVEL]

    # Use memory values to get corresponding names from pokemon_data
    names = [
        entry["name"]
        for entry in pokemon_data
        if entry.get("decimal") and int(entry["decimal"]) in memory_values
    ]

    # Create an initial dictionary with names as keys and levels as values
    party_dict = dict(zip(names, levels))

    return party_dict


def update_pokemon_level(pokemon_dict, pokemon_name, new_level):
    if pokemon_name in pokemon_dict:
        # Update the level for the specified Pokémon
        pokemon_dict[pokemon_name] = new_level
    else:
        # Add a new entry for the Pokémon
        pokemon_dict[pokemon_name] = new_level


# Returns dict of party pokemons' names, levels, and moves for printing to text file:
def pokemon_dict(pyboy):
    # Initialize a list of dictionaries for all 6 slots
    pokemon_info = [{"slot": str(i + 1), "name": "", "level": "0", "moves": []} for i in range(6)]
    # Iterate over each Pokémon slot
    for i in range(6):
        # Get the Pokémon and level for the current slot
        poke, lev = read_m(pyboy, POKE[i]), read_m(pyboy, LEVEL[i])
        # Convert the Pokémon's decimal value to hex and remove the '0x' prefix
        hex_value = hex(int(poke))[2:].upper()
        # Check if the hex value is in pokemon_data
        matching_pokemon = next(
            (entry for entry in pokemon_data if entry.get("hex") == hex_value), None
        )
        if matching_pokemon:
            # Update the Pokémon's name and level
            pokemon_info[i]["name"] = matching_pokemon["name"]
            pokemon_info[i]["level"] = str(lev)
            # Get the moves for the current Pokémon
            moves_addresses = [MOVE1[i], MOVE2[i], MOVE3[i], MOVE4[i]]
            pokemon_info[i]["moves"] = []  # Clear the moves for the current Pokémon
            for moves_address in moves_addresses:
                # Check each of the 4 possible moves
                move_value = read_m(pyboy, moves_address)
                if move_value != 0x00:
                    # Get the move information and add the move name to the Pokémon's moves
                    move_info = MOVES_DICT.get(move_value, {})
                    move_name = move_info.get("Move", "")
                    pokemon_info[i]["moves"].append(move_name)
    return pokemon_info


def position(pyboy):
    r_pos = read_m(pyboy, Y_POS_ADDR)
    c_pos = read_m(pyboy, X_POS_ADDR)
    map_n = read_m(pyboy, MAP_N_ADDR)
    if r_pos >= 443:
        r_pos = 444
    if r_pos <= 0:
        r_pos = 0
    if c_pos >= 443:
        c_pos = 444
    if c_pos <= 0:
        c_pos = 0
    if map_n > 247:
        map_n = 247
    if map_n < -1:
        map_n = -1
    return r_pos, c_pos, map_n


def relocate(pyboy, y, x):
    r_pos = write_mem(pyboy, Y_POS_ADDR, y)
    c_pos = write_mem(pyboy, X_POS_ADDR, x)
    r, c, n = position(pyboy)
    print(f"r, c, r={r}, c={c}, n={n}")
    return r_pos, c_pos


def read_party(pyboy):
    _, addr = symbol_lookup(pyboy, "wPartySpecies")
    party_length = pyboy.memory[symbol_lookup(pyboy, ("wPartyCount")[1])]
    return pyboy.memory[addr : addr + party_length]


def party(pyboy):
    # party = [read_m(pyboy, addr) for addr in PARTY_ADDR]
    party_size = read_m(pyboy, PARTY_SIZE_ADDR)
    party_levels = [x for x in [read_m(pyboy, addr) for addr in PARTY_LEVEL_ADDR] if x > 0]
    return party_size, party_levels  # [x for x in party_levels if x > 0]


def opponent(pyboy):
    return [read_m(pyboy, addr) for addr in OPPONENT_LEVEL_ADDR]


def oak_parcel(pyboy):
    return read_bit(pyboy, OAK_PARCEL_ADDR, 1)


def pokedex_obtained(pyboy):
    return read_bit(pyboy, OAK_POKEDEX_ADDR, 5)


def pokemon_seen(pyboy):
    seen_bytes = [read_m(pyboy, addr) for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])


def pokemon_caught(pyboy):
    caught_bytes = [read_m(pyboy, addr) for addr in CAUGHT_POKE_ADDR]
    return sum([bit_count(b) for b in caught_bytes])


# BET ADDED
def read_hp(pyboy, start):
    return 256 * read_m(pyboy, start) + read_m(pyboy, start + 1)


def read_hp_fraction(pyboy):
    party_size = read_m(pyboy, "wPartyCount")
    hp_sum = sum([read_short(pyboy, f"wPartyMon{i+1}HP") for i in range(party_size)])
    max_hp_sum = sum([read_short(pyboy, f"wPartyMon{i+1}MaxHP") for i in range(party_size)])
    max_hp_sum = max(max_hp_sum, 1)
    return hp_sum / max_hp_sum


def money(pyboy):
    return (
        100 * 100 * bcd(read_m(pyboy, MONEY_ADDR_1))
        + 100 * bcd(read_m(pyboy, MONEY_ADDR_100))
        + bcd(read_m(pyboy, MONEY_ADDR_10000))
    )


def badges(pyboy):
    badges = read_m(pyboy, BADGE_1_ADDR)
    return bit_count(badges)


def saved_bill(pyboy):
    """Restored Bill from his experiment"""
    return int(read_bit(pyboy, USED_CELL_SEPARATOR_ADDR, 3))


def events(pyboy):
    """Adds up all event flags, exclude museum ticket"""
    num_events = sum(bit_count(read_m(pyboy, i)) for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END))
    museum_ticket = int(read_bit(pyboy, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)


def talk_to_npc(pyboy):
    """
    Talk to NPC
    238 is text box arrow blink on
    127 is no text box arrow
    """
    return read_m(pyboy, TEXT_BOX_ARROW_BLINK)


def is_in_battle(pyboy):
    # D057
    # 0 not in battle
    # 1 wild battle
    # 2 trainer battle
    # -1 lost battle
    bflag = read_m(pyboy, BATTLE_FLAG)
    if bflag > 0:
        return True
    else:
        return False


def if_font_is_loaded(pyboy):
    return read_m(pyboy, IF_FONT_IS_LOADED)

    # get information for player


def player_direction(pyboy):
    return read_m(pyboy, PLAYER_DIRECTION)


def player_y(pyboy):
    return read_m(pyboy, PLAYER_Y)


def player_x(pyboy):
    return read_m(pyboy, PLAYER_X)


def map_n(pyboy):
    return read_m(pyboy, MAP_N_ADDR)


def npc_y(pyboy, npc_id):
    npc_id = npc_id * 0x10
    return read_m(pyboy, 0xC104 + npc_id)


def npc_x(pyboy, npc_id):
    npc_id = npc_id * 0x10
    return read_m(pyboy, 0xC106 + npc_id)


def sprites(pyboy):
    return read_m(pyboy, WNUMSPRITES)


def signs(pyboy):
    return read_m(pyboy, WNUMSIGNS)


def tree_tile(pyboy):
    return read_m(pyboy, WCUTTILE)


def rewardable_coords(glob_c, glob_r):
    include_conditions = [
        (80 >= glob_c >= 72) and (294 < glob_r <= 320),
        (69 < glob_c < 74) and (313 >= glob_r >= 295),
        (73 >= glob_c >= 72) and (220 <= glob_r <= 330),
        (75 >= glob_c >= 74) and (310 >= glob_r <= 319),
        # (glob_c >= 75 and glob_r <= 310),
        (81 >= glob_c >= 73) and (294 < glob_r <= 313),
        (73 <= glob_c <= 81) and (294 < glob_r <= 308),
        (80 >= glob_c >= 74) and (330 >= glob_r >= 284),
        (90 >= glob_c >= 89) and (336 >= glob_r >= 328),
        # New below
        # Viridian Pokemon Center
        (282 >= glob_r >= 277) and glob_c == 98,
        # Pewter Pokemon Center
        (173 <= glob_r <= 178) and glob_c == 42,
        # Route 4 Pokemon Center
        (131 <= glob_r <= 136) and glob_c == 132,
        (75 <= glob_c <= 76) and (271 < glob_r < 273),
        (82 >= glob_c >= 74) and (284 <= glob_r <= 302),
        (74 <= glob_c <= 76) and (284 >= glob_r >= 277),
        (76 >= glob_c >= 70) and (266 <= glob_r <= 277),
        (76 <= glob_c <= 78) and (274 >= glob_r >= 272),
        (74 >= glob_c >= 71) and (218 <= glob_r <= 266),
        (71 >= glob_c >= 67) and (218 <= glob_r <= 235),
        (106 >= glob_c >= 103) and (228 <= glob_r <= 244),
        (116 >= glob_c >= 106) and (228 <= glob_r <= 232),
        (116 >= glob_c >= 113) and (196 <= glob_r <= 232),
        (113 >= glob_c >= 89) and (208 >= glob_r >= 196),
        (97 >= glob_c >= 89) and (188 <= glob_r <= 214),
        (102 >= glob_c >= 97) and (189 <= glob_r <= 196),
        (89 <= glob_c <= 91) and (188 >= glob_r >= 181),
        (74 >= glob_c >= 67) and (164 <= glob_r <= 184),
        (68 >= glob_c >= 67) and (186 >= glob_r >= 184),
        (64 <= glob_c <= 71) and (151 <= glob_r <= 159),
        (71 <= glob_c <= 73) and (151 <= glob_r <= 156),
        (73 <= glob_c <= 74) and (151 <= glob_r <= 164),
        (103 <= glob_c <= 74) and (157 <= glob_r <= 156),
        (80 <= glob_c <= 111) and (155 <= glob_r <= 156),
        (111 <= glob_c <= 99) and (155 <= glob_r <= 150),
        (111 <= glob_c <= 154) and (150 <= glob_r <= 153),
        (138 <= glob_c <= 154) and (153 <= glob_r <= 160),
        (153 <= glob_c <= 154) and (153 <= glob_r <= 154),
        (143 <= glob_c <= 144) and (153 <= glob_r <= 154),
        (154 <= glob_c <= 158) and (134 <= glob_r <= 145),
        (152 <= glob_c <= 156) and (145 <= glob_r <= 150),
        (42 <= glob_c <= 43) and (173 <= glob_r <= 178),
        (158 <= glob_c <= 163) and (134 <= glob_r <= 135),
        (161 <= glob_c <= 163) and (114 <= glob_r <= 128),
        (163 <= glob_c <= 169) and (114 <= glob_r <= 115),
        (114 <= glob_c <= 169) and (167 <= glob_r <= 102),
        (169 <= glob_c <= 179) and (102 <= glob_r <= 103),
        (178 <= glob_c <= 179) and (102 <= glob_r <= 95),
        (178 <= glob_c <= 163) and (95 <= glob_r <= 96),
        (164 <= glob_c <= 163) and (110 <= glob_r <= 96),
        (163 <= glob_c <= 151) and (110 <= glob_r <= 109),
        (151 <= glob_c <= 154) and (101 <= glob_r <= 109),
        (151 <= glob_c <= 152) and (101 <= glob_r <= 97),
        (153 <= glob_c <= 154) and (97 <= glob_r <= 101),
        (151 <= glob_c <= 154) and (97 <= glob_r <= 98),
        (152 <= glob_c <= 155) and (69 <= glob_r <= 81),
        (155 <= glob_c <= 169) and (80 <= glob_r <= 81),
        (168 <= glob_c <= 184) and (39 <= glob_r <= 43),
        (183 <= glob_c <= 178) and (43 <= glob_r <= 51),
        (179 <= glob_c <= 183) and (48 <= glob_r <= 59),
        (179 <= glob_c <= 158) and (59 <= glob_r <= 57),
        (158 <= glob_c <= 161) and (57 <= glob_r <= 30),
        (158 <= glob_c <= 150) and (30 <= glob_r <= 31),
        (153 <= glob_c <= 150) and (34 <= glob_r <= 31),
        (168 <= glob_c <= 254) and (134 <= glob_r <= 140),
        (282 >= glob_r >= 277) and (436 >= glob_c >= 0),  # Include Viridian Pokecenter everywhere
        (173 <= glob_r <= 178) and (436 >= glob_c >= 0),  # Include Pewter Pokecenter everywhere
        (131 <= glob_r <= 136) and (436 >= glob_c >= 0),  # Include Route 4 Pokecenter everywhere
        (137 <= glob_c <= 197) and (82 <= glob_r <= 142),  # Mt Moon Route 3
        (137 <= glob_c <= 187) and (53 <= glob_r <= 103),  # Mt Moon B1F
        (137 <= glob_c <= 197) and (16 <= glob_r <= 66),  # Mt Moon B2F
        (137 <= glob_c <= 436) and (82 <= glob_r <= 444),  # Most of the rest of map after Mt Moon
        # (0 <= glob_c <= 436) and (0 <= glob_r <= 444),  # Whole map included
    ]
    return any(include_conditions)


def random_pokemon():
    # Generate a random number between 1 and 190 inclusive
    random_decimal = random.randint(1, 190)
    # Find the pokemon with the matching decimal value
    matching_pokemon = next(
        (entry["name"] for entry in pokemon_data if int(entry.get("decimal")) == random_decimal),
        None,
    )
    if matching_pokemon is None:
        # raise ValueError(f"No pokemon found with decimal value {random_decimal}")
        matching_pokemon = "Magikarp"
    # Print the name of the pokemon
    # print(f"Random Pokemon: {matching_pokemon}")
    return matching_pokemon


def read_bit(pyboy, addr: str | int, bit: int) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bool(int(read_m(pyboy, addr)) & (1 << bit))


def symbol_lookup(pyboy, symbol: str) -> tuple[int, int]:
    return pyboy.symbol_lookup(symbol)


@staticmethod
def set_bit(value, bit):
    return value | (1 << bit)


def mem_val(pyboy, addr):
    mem = read_m(pyboy, addr)
    return mem


def write_mem(pyboy, addr, value):
    mem = pyboy.memory[addr] = value
    return mem


def ss_anne_appeared(pyboy):
    """
    D803 - A bunch of bits that do different things
    """
    return read_m(pyboy, SS_ANNE)


def got_hm01(pyboy):
    return read_bit(pyboy, SS_ANNE, 0)


def rubbed_captains_back(pyboy):
    return read_bit(pyboy, SS_ANNE, 1)


def ss_anne_left(pyboy):
    return read_bit(pyboy, SS_ANNE, 2)


def walked_past_guard_after_ss_anne_left(pyboy):
    return read_bit(pyboy, SS_ANNE, 3)


def started_walking_out_of_dock(pyboy):
    return read_bit(pyboy, SS_ANNE, 4)


def walked_out_of_dock(pyboy):
    return read_bit(pyboy, SS_ANNE, 5)


def used_cut(pyboy):
    return read_m(pyboy, WCUTTILE)


def get_hm_count(pyboy) -> int:
    return len(HM_ITEM_IDS.intersection(get_items_in_bag(pyboy)))


def get_items_in_bag(pyboy, one_indexed=0):
    first_item = 0xD31E
    item_ids = []
    for i in range(0, 20, 2):
        item_id = read_m(pyboy, first_item + i)
        if item_id == 0 or item_id == 0xFF:
            break
        item_ids.append(item_id + one_indexed)
    return item_ids


def get_items_names(pyboy, one_indexed=0):
    first_item = 0xD31E
    item_names = []
    for i in range(0, 20, 2):
        item_id = read_m(pyboy, first_item + i)
        if item_id == 0 or item_id == 0xFF:
            break
        item_id_key = item_id + one_indexed
        item_name = data.items_dict.get(item_id_key, {}).get("Item", f"Unknown Item {item_id_key}")
        item_names.append(item_name)
    return item_names


def bill_capt(pyboy):
    met_bill = 5 * int(read_bit(pyboy, 0xD7F1, 0))
    used_cell_separator_on_bill = 5 * int(read_bit(pyboy, 0xD7F2, 3))
    ss_ticket = 5 * int(read_bit(pyboy, 0xD7F2, 4))
    met_bill_2 = 5 * int(read_bit(pyboy, 0xD7F2, 5))
    bill_said_use_cell_separator = 5 * int(read_bit(pyboy, 0xD7F2, 6))
    left_bills_house_after_helping = 5 * int(read_bit(pyboy, 0xD7F2, 7))
    got_hm01 = 5 * int(read_bit(pyboy, 0xD803, 0))
    rubbed_captains_back = 5 * int(read_bit(pyboy, 0xD803, 1))
    return sum(
        [
            met_bill,
            used_cell_separator_on_bill,
            ss_ticket,
            met_bill_2,
            bill_said_use_cell_separator,
            left_bills_house_after_helping,
            got_hm01,
            rubbed_captains_back,
        ]
    )


def read_ram_m(pyboy, addr) -> int:
    return read_m(pyboy, addr.value)


def read_ram_bit(pyboy, addr, bit: int) -> bool:
    return bin(256 + read_m(pyboy, addr.value))[-bit - 1] == "1"


def trash_can_memory(pyboy):
    return mem_val(pyboy, 0xCD5B)


def write_hp_for_first_pokemon(pyboy, new_hp, new_max_hp):
    """Write new HP value for the first party Pokémon."""
    # HP address for the first party Pokémon
    hp_addr = HP_ADDR[0]
    max_hp_addr = MAX_HP_ADDR[0]
    # Break down the new_hp value into two bytes
    hp_high = new_hp // 256  # Get the high byte
    hp_low = new_hp % 256  # Get the low byte
    max_hp_high = new_max_hp // 256  # Get the high byte
    max_hp_low = new_max_hp % 256  # Get the low byte
    # Write the high byte and low byte to the corresponding memory addresses
    write_mem(pyboy, hp_addr, hp_high)
    write_mem(pyboy, hp_addr + 1, hp_low)
    write_mem(pyboy, max_hp_addr, max_hp_high)
    write_mem(pyboy, max_hp_addr + 1, max_hp_low)
    # print(f"Set Max HP for the first party Pokémon to {new_max_hp}")
    # print(f"Set HP for the first party Pokémon to {new_hp}")


def update_party_hp_to_max(pyboy):
    """
    Update the HP of all party Pokémon to match their Max HP.
    """
    for i in range(len(CHP)):
        # Read Max HP
        max_hp = read_uint16(pyboy, MAX_HP_ADDR[i])
        # Calculate high and low bytes for Max HP to set as current HP
        hp_high = max_hp // 256
        hp_low = max_hp % 256
        # Update current HP to match Max HP
        write_mem(pyboy, CHP[i], hp_high)
        write_mem(pyboy, CHP[i] + 1, hp_low)
        # print(f"Updated Pokémon {i+1}: HP set to Max HP of {max_hp}.")


def restore_party_move_pp(pyboy):
    """
    Restores the PP of all moves for the party Pokémon based on MOVES_DICT data.
    """
    try:
        for i in range(len(MOVE1)):  # Assuming same length for MOVE1 to MOVE4
            moves_ids = [
                mem_val(pyboy, move_addr) for move_addr in [MOVE1[i], MOVE2[i], MOVE3[i], MOVE4[i]]
            ]

            for j, move_id in enumerate(moves_ids):
                if move_id in MOVES_DICT:
                    try:
                        # Fetch the move's max PP
                        max_pp = MOVES_DICT[move_id]["PP"]

                        # Determine the corresponding PP address based on the move slot
                        pp_addr = [MOVE1PP[i], MOVE2PP[i], MOVE3PP[i], MOVE4PP[i]][j]

                        # Restore the move's PP
                        write_mem(pyboy, pp_addr, max_pp)
                        # print(f"Restored PP for {MOVES_DICT[move_id]['Move']} to {max_pp}.")
                    except Exception as e:
                        # print(f"Error: {e}")
                        logging.info(f'error in ram_map.py restore_party_move_pp: {e}')
                        pass
                else:
                    pass
                    # print(f"Move ID {move_id} not found in MOVES_DICT.")
    except Exception as e:
        # print(f"Error: {e}")
        logging.info(f'Whole function error in ram_map.py restore_party_move_pp: {e}')
        pass

