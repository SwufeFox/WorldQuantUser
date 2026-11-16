"""A REGULAR simulation must not carry the selection settings.

``OPTIONS /simulations`` lists ``selectionHandling`` and ``selectionLimit`` as
required, so the capability preflight asks every spec for them. The POST
contract then answers 400 - "Selection handling NON_NAN is not allowed for
simulation type REGULAR" - which left the tool unable to submit anything at
all. The payload builder drops them where the simulation type is known.
"""

from __future__ import annotations

import unittest

import machine_lib as ml


class RegularPayloadDropsSelectionSettings(unittest.TestCase):
    def test_selection_settings_are_not_sent(self):
        settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "testPeriod": "P0Y",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "selectionHandling": "NON_NAN",
            "selectionLimit": 1000,
            "language": "FASTEXPR",
            "visualization": False,
        }

        payload = ml._single_simulation_data("rank(close)", 4, settings)

        self.assertEqual(payload["type"], "REGULAR")
        self.assertNotIn("selectionHandling", payload["settings"])
        self.assertNotIn("selectionLimit", payload["settings"])
        # Everything else must survive untouched.
        for key in ("region", "universe", "delay", "neutralization",
                    "unitHandling", "nanHandling", "language"):
            self.assertEqual(payload["settings"][key], settings[key])
        self.assertEqual(payload["settings"]["decay"], 4)
        self.assertEqual(payload["regular"], "rank(close)")

    def test_other_settings_are_not_dropped(self):
        payload = ml._single_simulation_data("rank(close)", 6,
                                             {"region": "USA", "decay": 99})
        self.assertEqual(payload["settings"]["region"], "USA")
        self.assertEqual(payload["settings"]["decay"], 6)


if __name__ == "__main__":
    unittest.main()