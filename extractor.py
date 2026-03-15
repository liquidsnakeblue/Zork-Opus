"""
Game state extractor - pure Z-machine data, no LLM calls.
Uses Jericho object tree for inventory, location, objects, exits, score.
"""

from typing import Optional, List
from pydantic import BaseModel
from game_interface import JerichoInterface


class ExtractedState(BaseModel):
    """All data comes from Z-machine - no LLM parsing needed."""
    location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    inventory: List[str]
    score: Optional[int] = None
    moves: Optional[int] = None


# Known Zork character keywords for NPC detection
_CHARACTER_KEYWORDS = ["thief", "troll", "cyclops", "bat", "grue", "ghost", "spirit", "demon"]


class Extractor:
    """Pure Jericho extractor - no LLM dependency."""

    def __init__(self, jericho: JerichoInterface, config=None, logger=None, episode_id: str = "unknown"):
        self.jericho = jericho
        self.logger = logger
        self.episode_id = episode_id

    def extract(self, game_text: str = "") -> Optional[ExtractedState]:
        try:
            loc = self.jericho.get_location()
            loc_name = loc.name if loc else "Unknown"
            inv = self.jericho.get_inventory()
            score, _ = self.jericho.get_score()
            exits = self.jericho.get_valid_exits()

            # Visible objects and characters from object tree
            visible = self.jericho.get_visible_objects()
            objects = [o.name for o in visible if o.name.strip() and o.name.lower() != "cretin"]
            characters = [o.name for o in visible
                         if any(k in o.name.lower() for k in _CHARACTER_KEYWORDS)]

            return ExtractedState(
                location_name=loc_name, exits=exits, visible_objects=objects,
                visible_characters=characters, inventory=inv, score=score,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Extraction failed: {e}")
            try:
                loc = self.jericho.get_location()
                name = loc.name if loc else "Extraction Failed"
            except Exception:
                name = "Extraction Failed"
            return ExtractedState(location_name=name, exits=[], visible_objects=[],
                                  visible_characters=[], inventory=[])

    def get_clean_text(self, game_text: str) -> str:
        return game_text

    def update_episode_id(self, eid: str):
        self.episode_id = eid
