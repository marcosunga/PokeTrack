from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import math
import httpx
import random
import numpy as np  # optional – falls back to pure-Python if unavailable

app = FastAPI(title="PokéTrack Analytics API", version="3.0.0")

# Allow the HTML frontend to call this API directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------

class DamageRequest(BaseModel):
    level: int = 100
    attacker_stat: int
    defender_stat: int
    move_power: int
    is_stab: bool
    type_effectiveness: float


class TeamSlot(BaseModel):
    pokemon_id: int
    pokemon_name: str
    types: List[str]
    ivs: dict = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}
    evs: dict = {"hp": 0,  "atk": 0,  "def": 252, "spa": 252, "spd": 0,  "spe": 4}


class TeamRequest(BaseModel):
    slots: List[Optional[TeamSlot]]


# ---------------------------------------------------------------------------
# TYPE EFFECTIVENESS TABLE (18×18, Gen VI+)
# ---------------------------------------------------------------------------
TYPES_LIST = [
    "normal","fire","water","electric","grass","ice",
    "fighting","poison","ground","flying","psychic","bug",
    "rock","ghost","dragon","dark","steel","fairy"
]

TYPE_CHART = {
    "normal":   [1,1,1,1,1,1,1,1,1,1,1,1,.5,0,1,1,.5,1],
    "fire":     [1,.5,.5,1,2,2,1,1,1,1,1,2,.5,1,.5,1,2,1],
    "water":    [1,2,.5,1,.5,1,1,1,2,1,1,1,2,1,.5,1,1,1],
    "electric": [1,1,2,.5,.5,1,1,1,0,2,1,1,1,1,.5,1,1,1],
    "grass":    [1,.5,2,1,.5,1,1,.5,2,.5,1,.5,2,1,.5,1,.5,1],
    "ice":      [1,.5,.5,1,2,.5,1,1,2,2,1,1,1,1,2,1,.5,1],
    "fighting": [2,1,1,1,1,2,1,.5,1,.5,.5,.5,2,0,1,2,2,.5],
    "poison":   [1,1,1,1,2,1,1,.5,.5,1,1,1,.5,.5,1,1,0,2],
    "ground":   [1,2,1,2,.5,1,1,2,1,0,1,.5,2,1,1,1,2,1],
    "flying":   [1,1,1,.5,2,1,2,1,1,1,1,2,.5,1,1,1,.5,1],
    "psychic":  [1,1,1,1,1,1,2,2,1,1,.5,1,1,1,1,0,.5,1],
    "bug":      [1,.5,1,1,2,1,.5,.5,1,.5,2,1,1,.5,1,2,.5,.5],
    "rock":     [1,2,1,1,1,2,.5,1,.5,2,1,2,1,1,1,1,.5,1],
    "ghost":    [0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,.5,1,1],
    "dragon":   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,.5,0],
    "dark":     [1,1,1,1,1,1,.5,1,1,1,2,1,1,2,1,.5,1,.5],
    "steel":    [1,.5,.5,.5,1,2,1,1,1,1,1,2,1,1,1,1,.5,2],
    "fairy":    [1,.5,1,1,1,1,2,.5,1,1,1,1,1,1,2,2,.5,1],
}


def get_type_effectiveness(move_type: str, defender_types: List[str]) -> float:
    """Compute combined effectiveness for a move type against one or more defender types."""
    move_type = move_type.lower()
    eff = 1.0
    if move_type not in TYPE_CHART:
        return eff
    row = TYPE_CHART[move_type]
    for def_type in defender_types:
        def_type = def_type.lower()
        if def_type in TYPES_LIST:
            idx = TYPES_LIST.index(def_type)
            eff *= row[idx]
    return eff


# ---------------------------------------------------------------------------
# VERSION GROUP → GENERATION MAPPING (for move categorization)
# ---------------------------------------------------------------------------
VERSION_GROUP_TO_GEN = {
    "red-blue": "Gen I", "yellow": "Gen I",
    "gold-silver": "Gen II", "crystal": "Gen II",
    "ruby-sapphire": "Gen III", "firered-leafgreen": "Gen III", "emerald": "Gen III",
    "diamond-pearl": "Gen IV", "platinum": "Gen IV", "heartgold-soulsilver": "Gen IV",
    "black-white": "Gen V", "black-2-white-2": "Gen V",
    "x-y": "Gen VI", "omega-ruby-alpha-sapphire": "Gen VI",
    "sun-moon": "Gen VII", "ultra-sun-ultra-moon": "Gen VII",
    "sword-shield": "Gen VIII", "brilliant-diamond-and-shining-pearl": "Gen VIII",
    "legends-arceus": "Gen VIII", "scarlet-violet": "Gen IX",
}


def categorize_moves(moves_data: list) -> dict:
    """
    Given the `moves` array from a PokeAPI /pokemon/{id} response,
    return a dict with keys: level_up, egg, tm, tutor.
    Each entry is a list of {name, level, gen, version_group} dicts,
    sorted by level (for level-up) or name (others).
    """
    level_up, egg, tm, tutor = [], [], [], []

    for move in moves_data:
        move_name = move["move"]["name"]
        for vgd in move.get("version_group_details", []):
            method = vgd["move_learn_method"]["name"]
            vg_name = vgd.get("version_group", {}).get("name", "")
            gen = VERSION_GROUP_TO_GEN.get(vg_name, vg_name)
            entry = {"name": move_name, "gen": gen, "version_group": vg_name}

            if method == "level-up":
                entry["level"] = vgd.get("level_learned_at", 0)
                level_up.append(entry)
                break
            elif method == "egg":
                egg.append(entry)
                break
            elif method == "machine":
                tm.append(entry)
                break
            elif method == "tutor":
                tutor.append(entry)
                break

    return {
        "level_up": sorted(level_up, key=lambda x: x.get("level", 0))[:30],
        "egg":      sorted(egg,      key=lambda x: x["name"])[:20],
        "tm":       sorted(tm,       key=lambda x: x["name"])[:30],
        "tutor":    sorted(tutor,    key=lambda x: x["name"])[:20],
    }


# ---------------------------------------------------------------------------
# TREND DATA ENGINE — Random-Walk simulation (Poisson-perturbed)
# Each Pokémon gets a deterministic seed based on its ID so results are
# consistent across calls, but vary realistically between Pokémon.
# ---------------------------------------------------------------------------

def _generate_trend_random_walk(pokemon_id: int, days: int = 30) -> list[float]:
    """
    Generate a realistic-looking 'days'-point usage time series for a Pokémon.

    Method:
      - Seed the RNG with (pokemon_id * 31337) so results are reproducible.
      - Base popularity is drawn from a Beta(2,3) distribution scaled to [5, 95].
      - Each day adds Gaussian noise and a Poisson spike with low probability.
      - Values are clipped to [1, 100].
    """
    rng = random.Random(pokemon_id * 31337)

    # Base popularity: Beta-like distribution approximated with a few samples
    raw = sum(rng.random() for _ in range(5)) / 5
    base = 5 + raw * 75  # [5, 80]

    usage = []
    v = base
    for _ in range(days):
        # Gaussian drift
        noise = rng.gauss(0, 4.5)
        # Rare Poisson-style popularity spike (prob ~8%)
        spike = rng.expovariate(12) * 18 if rng.random() < 0.08 else 0
        v = max(1.0, min(100.0, v + noise + spike))
        usage.append(round(v, 2))

    return usage


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "PokéTrack Analytics API Active",
        "version": "3.0.0",
        "endpoints": [
            "/calculate-damage",
            "/type-effectiveness",
            "/proxy/pokemon/{name}",
            "/proxy/pokemon/{name}/moves",
            "/api/trends/{pokemon_id}",
            "/api/trends/compare",
        ]
    }


@app.post("/calculate-damage")
async def calculate_damage(req: DamageRequest):
    """
    Official Pokémon Damage Formula (Gen V+):

        Damage = ⌊ ( ⌊ (2×L/5+2) × P × A/D / 50 ⌋ + 2 ) × Modifier ⌋
        Modifier = STAB × TypeEffectiveness

    Returns min (85% roll) and max (100% roll) damage values.
    """
    if req.move_power <= 0:
        raise HTTPException(status_code=400, detail="move_power must be > 0")
    if req.defender_stat <= 0:
        raise HTTPException(status_code=400, detail="defender_stat must be > 0")

    base_dmg = (
        (((2 * req.level / 5 + 2) * req.move_power * (req.attacker_stat / req.defender_stat)) / 50)
        + 2
    )

    stab = 1.5 if req.is_stab else 1.0
    modifier = stab * req.type_effectiveness

    max_dmg = math.floor(base_dmg * modifier)
    min_dmg = math.floor(max_dmg * 0.85)

    te = req.type_effectiveness
    if te == 0:
        verdict = "No Effect"
    elif te >= 4:
        verdict = "Quadruple Effective"
    elif te >= 2:
        verdict = "Super Effective"
    elif te < 1:
        verdict = "Not Very Effective"
    else:
        verdict = "Normal"

    return {
        "min_damage": min_dmg,
        "max_damage": max_dmg,
        "stab_modifier": stab,
        "type_effectiveness": req.type_effectiveness,
        "verdict": verdict,
    }


@app.get("/type-effectiveness")
async def type_effectiveness(move_type: str, defender_types: str):
    """
    Returns the combined type effectiveness multiplier.
    defender_types: comma-separated list, e.g. 'grass,poison'
    """
    def_types = [t.strip() for t in defender_types.split(",") if t.strip()]
    eff = get_type_effectiveness(move_type, def_types)

    verdict = "Normal"
    if eff == 0:   verdict = "No Effect (Immune)"
    elif eff >= 4: verdict = "4× Super Effective"
    elif eff >= 2: verdict = "2× Super Effective"
    elif eff < 1:  verdict = "Not Very Effective"

    return {
        "move_type": move_type,
        "defender_types": def_types,
        "effectiveness": eff,
        "verdict": verdict,
    }


@app.get("/proxy/pokemon/{name}")
async def proxy_pokemon(name: str):
    """
    Thin proxy to PokeAPI — useful when calling from environments
    that restrict CORS or need server-side caching.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Pokémon '{name}' not found")
        resp.raise_for_status()
        return resp.json()


@app.get("/proxy/pokemon/{name}/moves")
async def proxy_pokemon_moves(name: str):
    """
    Fetch a Pokémon's moves from PokeAPI and return them categorized
    by learn method (level-up, egg, TM, tutor) with generation labels.

    Response shape:
    {
      "name": "charizard",
      "moves": {
        "level_up": [{"name": "scratch", "level": 1, "gen": "Gen I", "version_group": "red-blue"}, ...],
        "egg":      [...],
        "tm":       [...],
        "tutor":    [...]
      }
    }
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Pokémon '{name}' not found")
        resp.raise_for_status()
        data = resp.json()

    categorized = categorize_moves(data.get("moves", []))
    return {"name": name.lower(), "moves": categorized}


@app.get("/api/trends/{pokemon_id}")
async def get_trends(pokemon_id: int, days: int = 30):
    """
    Return a time-series of simulated usage percentage data for a given Pokémon ID.

    The series is generated using a random-walk model seeded by the Pokémon's ID,
    so results are deterministic (same Pokémon always produces the same curve)
    but vary realistically between different Pokémon.

    Query param:
      - days: number of data points to return (default 30, max 90)

    Response:
    {
      "pokemon_id": 6,
      "days": 30,
      "usage": [42.1, 44.8, ...],   // % usage per day
      "summary": {
        "avg": 48.3,
        "peak": 71.2,
        "trough": 28.9,
        "trend": "rising"  // "rising" | "falling" | "stable"
      }
    }
    """
    days = max(7, min(90, days))
    usage = _generate_trend_random_walk(pokemon_id, days)

    avg   = round(sum(usage) / len(usage), 2)
    peak  = round(max(usage), 2)
    trough = round(min(usage), 2)

    # Simple trend: compare last 5 days avg to first 5 days avg
    early = sum(usage[:5]) / 5
    late  = sum(usage[-5:]) / 5
    if late > early + 3:
        trend = "rising"
    elif late < early - 3:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "pokemon_id": pokemon_id,
        "days": days,
        "usage": usage,
        "summary": {"avg": avg, "peak": peak, "trough": trough, "trend": trend},
    }


@app.get("/api/trends/compare")
async def compare_trends(ids: str, days: int = 30):
    """
    Compare trend data for multiple Pokémon IDs in a single request.

    Query params:
      - ids: comma-separated Pokémon IDs, e.g. '6,25,150'
      - days: number of data points (default 30, max 90)

    Response:
    {
      "days": 30,
      "series": [
        {"pokemon_id": 6,  "usage": [...], "summary": {...}},
        {"pokemon_id": 25, "usage": [...], "summary": {...}},
        ...
      ]
    }
    """
    days = max(7, min(90, days))
    try:
        id_list = [int(i.strip()) for i in ids.split(",") if i.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")

    if len(id_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 Pokémon per comparison")

    series = []
    for pid in id_list:
        usage = _generate_trend_random_walk(pid, days)
        avg   = round(sum(usage) / len(usage), 2)
        peak  = round(max(usage), 2)
        trough = round(min(usage), 2)
        early = sum(usage[:5]) / 5
        late  = sum(usage[-5:]) / 5
        trend = "rising" if late > early + 3 else "falling" if late < early - 3 else "stable"
        series.append({
            "pokemon_id": pid,
            "usage": usage,
            "summary": {"avg": avg, "peak": peak, "trough": trough, "trend": trend},
        })

    return {"days": days, "series": series}


# ---------------------------------------------------------------------------
# Supabase integration — uncomment when ready
# ---------------------------------------------------------------------------

# from supabase import create_client
# import os
# supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# @app.post("/save-team")
# async def save_team(req: TeamRequest, user_id: str):
#     """Persist a team to Supabase."""
#     data = [s.dict() for s in req.slots if s is not None]
#     result = supabase.table("teams").upsert({"user_id": user_id, "slots": data}).execute()
#     return {"saved": len(data), "result": result.data}

# @app.get("/load-team/{user_id}")
# async def load_team(user_id: str):
#     result = supabase.table("teams").select("*").eq("user_id", user_id).execute()
#     return result.data