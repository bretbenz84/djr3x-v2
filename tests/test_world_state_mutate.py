"""Tests for WorldState.mutate — the atomic read-modify-write that closes the
lost-update race on shared fields like "people".

The concurrency test fails on the old get()+update() pattern (most appends are
lost) and passes with mutate(), so it pins the fix in place.
"""

import threading
import unittest

from world_state import world_state


class WorldStateMutateTest(unittest.TestCase):
    def setUp(self):
        self._saved_people = world_state.get("people")
        world_state.update("people", [])

    def tearDown(self):
        world_state.update("people", self._saved_people)

    def test_returns_new_value_and_persists(self):
        out = world_state.mutate("people", lambda p: p + [{"id": "x"}])
        self.assertEqual(out, [{"id": "x"}])
        self.assertEqual(world_state.get("people"), [{"id": "x"}])

    def test_none_means_no_change(self):
        world_state.update("people", [{"id": "a"}])
        out = world_state.mutate("people", lambda p: None)
        self.assertEqual(out, [{"id": "a"}])
        self.assertEqual(world_state.get("people"), [{"id": "a"}])

    def test_unknown_field_raises(self):
        with self.assertRaises(KeyError):
            world_state.mutate("not_a_field", lambda v: v)

    def test_exception_leaves_field_unchanged(self):
        # fn mutating its copy and then raising must not persist a partial write.
        world_state.update("people", [{"id": "keep"}])

        def boom(people):
            people.append({"id": "partial"})
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            world_state.mutate("people", boom)
        self.assertEqual(world_state.get("people"), [{"id": "keep"}])

    def test_no_lost_updates_under_concurrency(self):
        world_state.update("people", [])
        thread_count, per_thread = 8, 40

        def worker(tag):
            for i in range(per_thread):
                world_state.mutate("people", lambda p: p + [{"tag": tag, "i": i}])

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = world_state.get("people")
        # Every append from every thread landed — none silently overwritten.
        self.assertEqual(len(result), thread_count * per_thread)
        seen = {(p["tag"], p["i"]) for p in result}
        self.assertEqual(len(seen), thread_count * per_thread)

    def test_concurrent_field_writes_do_not_clobber(self):
        # Two writers touch different fields of the same slot concurrently; both
        # edits must survive (the identity-flicker scenario in miniature).
        world_state.update("people", [{"id": "p1", "person_db_id": None, "pose": None}])

        def set_identity():
            for _ in range(200):
                def _apply(people):
                    if people:
                        people[0]["person_db_id"] = 42
                        return people
                    return None
                world_state.mutate("people", _apply)

        def set_pose():
            for _ in range(200):
                def _apply(people):
                    if people:
                        people[0]["pose"] = "leaning_in"
                        return people
                    return None
                world_state.mutate("people", _apply)

        a = threading.Thread(target=set_identity)
        b = threading.Thread(target=set_pose)
        a.start(); b.start(); a.join(); b.join()

        slot = world_state.get("people")[0]
        self.assertEqual(slot["person_db_id"], 42)      # identity not reverted
        self.assertEqual(slot["pose"], "leaning_in")    # pose not reverted


if __name__ == "__main__":
    unittest.main()
