"""
Scenario tests for the learning-loop refactor: typed predicates, target
validation, fog-of-war treasure tracking, milestone-aware turn selection,
trial judgment, event-driven replanning, and LLM-call gating.

Run:  .venv/bin/python -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from state import GameState, Objective
from objectives import ObjectiveManager, ObjectiveDefinition
from knowledge import _format_turns_selective
from evidence import TrialLog
from treasures import TreasureRegistry, DEPOSITED, CARRIED, UNKNOWN, LOCATED
from orchestrator import Orchestrator


def make_config(**overrides):
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_map_manager(rooms):
    game_map = SimpleNamespace(room_names=dict(rooms), rooms=dict(rooms))
    return SimpleNamespace(game_map=game_map)


def make_objectives(gs=None, rooms=None, treasures=None):
    gs = gs or GameState()
    mgr = ObjectiveManager(
        make_config(), gs,
        map_manager=make_map_manager(rooms or {}),
        llm_client=object(),  # never called in these tests
        treasure_registry=treasures,
    )
    return mgr, gs


class TestPredicates(unittest.TestCase):
    def _obj(self, pred, **kw):
        return Objective(id="A001", category="action", name="t", text="t",
                         completion_condition="t", completion_predicate=pred, **kw)

    def test_inventory_contains(self):
        mgr, gs = make_objectives()
        gs.current_inventory = ["brass lantern", "elvish sword"]
        self.assertTrue(mgr._eval_predicate(
            self._obj(None), {"type": "inventory_contains", "item": "sword"}))
        self.assertFalse(mgr._eval_predicate(
            self._obj(None), {"type": "inventory_contains", "item": "egg"}))

    def test_room_id_equals(self):
        mgr, gs = make_objectives()
        gs.current_room_id = 107
        self.assertTrue(mgr._eval_predicate(
            self._obj(None), {"type": "room_id_equals", "room_id": 107}))
        self.assertFalse(mgr._eval_predicate(
            self._obj(None), {"type": "room_id_equals", "room_id": 41}))

    def test_score_delta_at_least(self):
        mgr, gs = make_objectives()
        gs.previous_score = 45
        obj = self._obj(None, created_score=40)
        self.assertTrue(mgr._eval_predicate(
            obj, {"type": "score_delta_at_least", "amount": 5}))
        self.assertFalse(mgr._eval_predicate(
            obj, {"type": "score_delta_at_least", "amount": 6}))

    def test_new_rooms_since_created(self):
        mgr, gs = make_objectives(rooms={i: f"R{i}" for i in range(10)})
        obj = self._obj(None, created_room_count=7)
        self.assertTrue(mgr._eval_predicate(
            obj, {"type": "new_rooms_since_created", "count": 3}))
        self.assertFalse(mgr._eval_predicate(
            obj, {"type": "new_rooms_since_created", "count": 4}))

    def test_trophy_contains_via_gs(self):
        mgr, gs = make_objectives()
        gs.trophy_case = {"painting"}
        self.assertTrue(mgr._eval_predicate(
            self._obj(None), {"type": "trophy_contains", "item": "painting"}))

    def test_deterministic_completion_and_unblock(self):
        mgr, gs = make_objectives()
        gs.turn_count = 12
        gs.current_room_id = 107
        obj = self._obj({"type": "room_id_equals", "room_id": 107})
        obj.status = "blocked"  # reaching the goal completes even a blocked objective
        gs.objectives.append(obj)
        completed = mgr.check_completions_deterministic()
        self.assertEqual(completed, ["A001"])
        self.assertEqual(obj.status, "completed")
        self.assertEqual(obj.completed_turn, 12)
        self.assertIn("objective_completed", mgr._pending_events)

    def test_a038_regression_failed_action_does_not_complete(self):
        """The A038 failure mode: bolt fails → objective must NOT complete.
        With a typed predicate there is no prose for a reviewer to misread."""
        mgr, gs = make_objectives()
        gs.current_inventory = ["wrench"]  # bolt NOT turned: no state change
        obj = self._obj({"type": "score_delta_at_least", "amount": 1},
                        created_score=0)
        gs.objectives.append(obj)
        self.assertEqual(mgr.check_completions_deterministic(), [])
        self.assertEqual(obj.status, "pending")


class TestPredicateSanitization(unittest.TestCase):
    def test_valid_predicate_kept(self):
        d = ObjectiveDefinition(
            category="action", name="n", text="t", completion_condition="c",
            completion_predicate={"type": "inventory_contains", "item": "egg"})
        self.assertEqual(d.completion_predicate,
                         {"type": "inventory_contains", "item": "egg"})

    def test_unknown_type_dropped(self):
        d = ObjectiveDefinition(
            category="action", name="n", text="t", completion_condition="c",
            completion_predicate={"type": "world_flag_equals", "flag": "x"})
        self.assertIsNone(d.completion_predicate)

    def test_coerces_l_prefixed_room_id(self):
        d = ObjectiveDefinition(
            category="action", name="n", text="t", completion_condition="c",
            completion_predicate={"type": "room_id_equals", "room_id": "L107"})
        self.assertEqual(d.completion_predicate,
                         {"type": "room_id_equals", "room_id": 107})

    def test_missing_arg_dropped(self):
        d = ObjectiveDefinition(
            category="action", name="n", text="t", completion_condition="c",
            completion_predicate={"type": "inventory_contains"})
        self.assertIsNone(d.completion_predicate)


class TestTargetValidation(unittest.TestCase):
    ROOMS = {41: "East-West Passage", 107: "Round Room", 193: "Living Room",
             50: "Reservoir South", 100: "Reservoir"}

    def test_round_room_regression(self):
        """Gen 63 bug: 'Reach Round Room Hub' was assigned L41 (East-West
        Passage). The validator must correct it to L107 (Round Room)."""
        mgr, _ = make_objectives(rooms=self.ROOMS)
        target, note = mgr._resolve_target(
            "Reach Round Room Hub",
            "Navigate to the Round Room hub via East-West Passage.", 41)
        self.assertEqual(target, 107)
        self.assertIn("corrected", note)

    def test_matching_target_untouched(self):
        mgr, _ = make_objectives(rooms=self.ROOMS)
        target, note = mgr._resolve_target(
            "Explore East-West Passage", "Check the passage for exits.", 41)
        self.assertEqual(target, 41)
        self.assertIsNone(note)

    def test_longest_name_wins(self):
        mgr, _ = make_objectives(rooms=self.ROOMS)
        target, _ = mgr._resolve_target("Reach Reservoir South", "", 100)
        self.assertEqual(target, 50)

    def test_unknown_room_id_dropped(self):
        mgr, _ = make_objectives(rooms=self.ROOMS)
        target, note = mgr._resolve_target("Find the Rainbow", "", 999)
        self.assertIsNone(target)
        self.assertIn("dropped", note)

    def test_missing_target_resolved_from_wording(self):
        mgr, _ = make_objectives(rooms=self.ROOMS)
        target, note = mgr._resolve_target("Deposit egg", "Return to Living Room.", None)
        self.assertEqual(target, 193)
        self.assertIn("resolved", note)


class TestEventDrivenReasoner(unittest.TestCase):
    def test_min_gap_blocks_thrash(self):
        mgr, gs = make_objectives()
        gs.turn_count = 2
        gs.objective_update_turn = 1
        mgr.notify_event("score_change")
        self.assertFalse(mgr.should_run_reasoner())

    def test_event_triggers_after_min_gap(self):
        mgr, gs = make_objectives()
        gs.turn_count = 10
        gs.objective_update_turn = 5
        self.assertFalse(mgr.should_run_reasoner())  # no events, gap < max
        mgr.notify_event("theft")
        self.assertTrue(mgr.should_run_reasoner())
        self.assertIn("theft", mgr._trigger_reason)

    def test_max_gap_fallback(self):
        mgr, gs = make_objectives()
        gs.turn_count = 30
        gs.objective_update_turn = 5
        self.assertTrue(mgr.should_run_reasoner())  # 25-turn max gap
        self.assertIn("Max-gap", mgr._trigger_reason)


class TestBlockedObjectives(unittest.TestCase):
    def test_mark_blocked(self):
        mgr, gs = make_objectives()
        gs.objectives.append(Objective(
            id="E001", category="exploration", name="Reach Round Room",
            text="t", completion_condition="c", target_location_id=107))
        self.assertTrue(mgr.mark_blocked("E001", "no path from L88"))
        obj = gs.get_objective("E001")
        self.assertEqual(obj.status, "blocked")
        self.assertEqual(obj.blocked_reason, "no path from L88")
        self.assertNotIn(obj, gs.active_objectives)
        self.assertIn(obj, gs.blocked_objectives)
        self.assertIn("objective_blocked", mgr._pending_events)


class FakeZObject:
    def __init__(self, num, name, parent, touched=False):
        self.num, self.name, self.parent = num, name, parent
        self.attr = [0, 0, 0, 1 if touched else 0] + [0] * 28


class FakeJericho:
    """Object tree: player(1)@LivingRoom(193); case(2)@193; painting(10) in case;
    egg(11) inside nest(12) at Up a Tree(88); coffin(13) lying in room 55."""
    def __init__(self):
        self.objects = [
            FakeZObject(2, "trophy case", 193),
            FakeZObject(10, "painting", 2, touched=True),
            FakeZObject(12, "bird's nest", 88),
            FakeZObject(11, "jewel-encrusted egg", 12),
            FakeZObject(13, "gold coffin", 55),
            FakeZObject(20, "thief", 60),
        ]
        self.player = FakeZObject(1, "cretin", 193)

    def get_all_objects(self): return self.objects + [self.player]
    def get_player_object(self): return self.player
    def room_name(self, rid): return {55: "Egyptian Room", 88: "Up a Tree"}.get(rid)
    def check_attribute(self, obj, bit):
        return bool(obj.attr[bit]) if obj is not None and bit < len(obj.attr) else False


class TestTreasureRegistry(unittest.TestCase):
    def _registry(self, tmpdir):
        cfg = make_config(game_workdir=str(tmpdir))
        return TreasureRegistry(cfg)

    def test_fog_of_war_and_states(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            reg = self._registry(td)
            jericho = FakeJericho()
            reg.update(jericho, current_room_id=193, current_room_name="Living Room", turn=5)

            # painting is in the trophy case → deposited (and matched despite
            # "trophy case" also containing no treasure fragment)
            self.assertEqual(reg.episode_state["painting"]["status"], DEPOSITED)
            self.assertIn("painting", reg.deposited())
            self.assertTrue(reg.is_deposited("painting"))
            # egg is nested in an unvisited room and untouched → fog of war holds
            self.assertEqual(reg.episode_state["egg"]["status"], UNKNOWN)
            # coffin lies in a room the agent is NOT in → still unknown
            self.assertEqual(reg.episode_state["coffin"]["status"], UNKNOWN)

            # Agent walks into the Egyptian Room → coffin visibly on the floor
            reg.update(jericho, current_room_id=55, current_room_name="Egyptian Room", turn=9)
            self.assertEqual(reg.episode_state["coffin"]["status"], LOCATED)
            self.assertEqual(reg.episode_state["coffin"]["room_id"], 55)

    def test_carried_and_stolen_gating(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            reg = self._registry(td)
            jericho = FakeJericho()
            # coffin moves to player's inventory
            jericho.objects[4].parent = 1
            reg.update(jericho, 193, "Living Room", 3)
            self.assertEqual(reg.episode_state["coffin"]["status"], CARRIED)
            # thief steals it → stolen IS revealed (agent knew about it)
            jericho.objects[4].parent = 20
            reg.update(jericho, 193, "Living Room", 4)
            self.assertEqual(reg.episode_state["coffin"]["status"], "stolen")
            # but a treasure never seen that the thief holds stays unknown
            jericho.objects[3].parent = 20  # egg → thief
            reg.update(jericho, 193, "Living Room", 5)
            self.assertEqual(reg.episode_state["egg"]["status"], UNKNOWN)

    def test_reasoner_format_counts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            reg = self._registry(td)
            reg.update(FakeJericho(), 193, "Living Room", 1)
            text = reg.format_for_reasoner()
            self.assertIn("1/19 deposited", text)
            self.assertIn("Stone Barrow", text)


class TestTurnSelection(unittest.TestCase):
    def _acts(self, n, score_turns):
        acts, score = [], 0
        for i in range(1, n + 1):
            if i in score_turns:
                score += 5
            acts.append({"turn": i, "action": f"action {i}", "score": score,
                         "location": "Somewhere", "response": "x" * 100})
        return {"actions_and_responses": acts}

    def test_late_game_milestones_survive(self):
        """The old head-only slice dropped everything after ~turn 60. Late
        score milestones must now always be present."""
        data = self._acts(500, {5, 490})
        text = _format_turns_selective(data, budget=3000)
        self.assertIn("Turn 5:", text)     # early milestone kept
        self.assertIn("Turn 490:", text)   # LATE milestone kept
        self.assertIn("omitted", text)     # truncation is explicit, not silent
        self.assertLessEqual(len(text), 3000 + 500)  # markers may exceed slightly

    def test_recency_fill(self):
        data = self._acts(100, set())
        text = _format_turns_selective(data, budget=2000)
        self.assertIn("Turn 100:", text)   # most recent turns kept
        self.assertNotIn("Turn 1:", text)  # oldest uneventful dropped
        self.assertIn("omitted", text)

    def test_small_episode_untouched(self):
        data = self._acts(5, {2})
        text = _format_turns_selective(data, budget=15000)
        for i in range(1, 6):
            self.assertIn(f"Turn {i}:", text)
        self.assertNotIn("omitted", text)


class TestTrialJudgment(unittest.TestCase):
    def test_verdicts(self):
        self.assertTrue(TrialLog.judge("Taken.", 0, False, ["egg"]))
        self.assertTrue(TrialLog.judge("", 5, False, []))
        self.assertTrue(TrialLog.judge("", 0, True, []))
        self.assertFalse(TrialLog.judge("The bolt won't turn with your best effort.", 0, False, []))
        self.assertFalse(TrialLog.judge("You can't go that way.", 0, False, []))
        # "sluice gates open" — real success invisible to the heuristic → ambiguous, not failure
        self.assertIsNone(TrialLog.judge(
            "The sluice gates open and water pours through the dam.", 0, False, []))


class TestLLMGating(unittest.TestCase):
    def _orch(self, **cfg_overrides):
        o = object.__new__(Orchestrator)  # skip heavy __init__
        o.config = make_config(**cfg_overrides)
        o.gs = GameState()
        return o

    def test_boring_parser_noise_skipped(self):
        o = self._orch()
        z = {"score_delta": 0, "location_changed": False, "inventory_changed": False,
             "died": False, "first_visit": False}
        self.assertFalse(o._should_synthesize_memory(
            "north", "You can't go that way.", z))
        self.assertFalse(o._should_synthesize_memory("look", "West of House...", z))

    def test_substantive_failure_still_synthesizes(self):
        """'The bolt won't turn' is learning signal — must NOT be skipped."""
        o = self._orch()
        z = {"score_delta": 0, "location_changed": False, "inventory_changed": False,
             "died": False, "first_visit": False}
        self.assertTrue(o._should_synthesize_memory(
            "turn bolt with wrench", "The bolt won't turn with your best effort.", z))

    def test_state_change_always_synthesizes(self):
        o = self._orch()
        z = {"score_delta": 5, "location_changed": False, "inventory_changed": False,
             "died": False, "first_visit": False}
        self.assertTrue(o._should_synthesize_memory("take egg", "Taken.", z))

    def test_flag_off_disables_gating(self):
        o = self._orch(memory_synthesis_skip_boring=False)
        z = {"score_delta": 0, "location_changed": False, "inventory_changed": False,
             "died": False, "first_visit": False}
        self.assertTrue(o._should_synthesize_memory("north", "You can't go that way.", z))

    def test_completion_review_gating(self):
        o = self._orch()
        o.gs.turn_count = 7  # not a sweep turn
        quiet = {"score_delta": 0, "location_changed": False,
                 "inventory_changed": False, "died": False}
        self.assertFalse(o._completion_review_due("You can't go that way.", quiet))
        self.assertTrue(o._completion_review_due(
            "A long and interesting response " * 10, quiet))
        o.gs.turn_count = 10  # periodic sweep still runs
        self.assertTrue(o._completion_review_due("short", quiet))


if __name__ == "__main__":
    unittest.main()
