from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import math
import httpx

app = FastAPI(title="PokéTrack Analytics API", version="2.0.0")

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
# Rows = attacking type, Cols = defending types (same order as TYPES_LIST)
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
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "PokéTrack Analytics API Active",
        "version": "2.0.0",
        "endpoints": ["/calculate-damage", "/type-effectiveness", "/proxy/pokemon/{name}"]
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

    # Core formula
    base_dmg = (
        (((2 * req.level / 5 + 2) * req.move_power * (req.attacker_stat / req.defender_stat)) / 50)
        + 2
    )

    stab = 1.5 if req.is_stab else 1.0
    modifier = stab * req.type_effectiveness

    max_dmg = math.floor(base_dmg * modifier)
    min_dmg = math.floor(max_dmg * 0.85)

    # Verdict
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


# ---------------------------------------------------------------------------
# Future Supabase integration (uncomment when ready)
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