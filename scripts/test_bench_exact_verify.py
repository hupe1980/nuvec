"""Negative controls for the serial-prefix check in the DFlash benchmark."""

import copy
import unittest

from bench_exact_verify import check_rounds


class RoundTests(unittest.TestCase):
    def test_rejections_and_short_tail(self):
        ref = [2, 3, 4, 5, 6, 7]
        row = {
            "ids": ref,
            "rounds": [
                {"start": 0, "draft": [2, 9, 9], "accepted": 1, "next": 3},
                {"start": 2, "draft": [4, 5, 9], "accepted": 2, "next": 6},
            ],
            "verifies": 2,
            "drafted": 6,
            "accepted": 3,
            "max_verify_rows": 4,
        }
        check_rounds(row, ref, 3)
        for mutate in (
            lambda rec: rec["rounds"][0].update(accepted=0),
            lambda rec: rec["rounds"][0].update(next=8),
            lambda rec: rec["rounds"][1].update(start=4),
            lambda rec: rec["rounds"].pop(),
            lambda rec: rec.update(accepted=4),
            lambda rec: rec.update(ids=ref[:-1]),
        ):
            corrupt = copy.deepcopy(row)
            mutate(corrupt)
            with self.assertRaises(ValueError):
                check_rounds(corrupt, ref, 3)


if __name__ == "__main__":
    unittest.main()
