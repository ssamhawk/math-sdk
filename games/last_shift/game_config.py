"""Static T4-C1 contract configuration for Last Shift.

This stage deliberately contains no reel weights, tuned RTP, or simulation
distributions. Those belong to T4-C2 after the rule skeleton is approved.
"""

from src.config.config import Config


class GameConfig(Config):
    """SDK-facing game constants without unvalidated production tuning."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_last_shift_initialized", False):
            return
        super().__init__()
        self.game_id = "last_shift"
        self.game_name = "last_shift"
        self.working_name = "Last Shift"
        self.provider_number = 0
        self.win_type = "scatter"
        self.construct_paths()

        self.num_reels = 6
        self.num_rows = [5] * self.num_reels
        self.include_padding = False
        self.regular_symbols = tuple("ABCDEFGH")
        self.special_symbols = {"wild": ["W"], "scatter": ["S"]}

        self.payout_scale = 100
        self.wincap = 20_000
        self.wincap_units = self.wincap * self.payout_scale
        self.minimum_scatter_pay_count = 8
        self.minimum_cargo_count = 2
        self.maximum_columns_loaded = 2
        self.initial_free_spins = 10
        self.retrigger_free_spins = 4

        # Provisional integer-unit paytable retained from the comparative
        # model. It is a mechanics fixture, not frozen T4-C2 tuning.
        count_pays_units = {
            8: 15,
            9: 20,
            10: 30,
            11: 45,
            12: 70,
            13: 100,
            14: 140,
            15: 200,
            16: 280,
            17: 400,
            18: 550,
            19: 750,
            20: 1_000,
            21: 1_400,
            22: 1_900,
            23: 2_600,
            24: 3_600,
            25: 5_000,
            26: 7_000,
            27: 10_000,
            28: 15_000,
            29: 22_500,
            30: 35_000,
        }
        symbol_values_units = {
            "A": 50,
            "B": 65,
            "C": 80,
            "D": 100,
            "E": 125,
            "F": 160,
            "G": 220,
            "H": 280,
        }
        self.paytable_units = {
            (count, symbol): (count_pay * symbol_value) // self.payout_scale
            for count, count_pay in count_pays_units.items()
            for symbol, symbol_value in symbol_values_units.items()
        }
        self.paytable = {
            key: units / self.payout_scale for key, units in self.paytable_units.items()
        }

        self.mode_definitions = {
            self.basegame_type: {"naturalBonus": True, "stateAcrossBets": False},
            self.freegame_type: {
                "initialSpins": self.initial_free_spins,
                "retriggerSpins": self.retrigger_free_spins,
                "stateAcrossFreeSpins": True,
            },
        }
        # Forced books are generated through a distinct path and must never be
        # sampled when measuring natural feature frequency in T4-C2.
        self.outcome_paths = {
            "natural": ("basegame", "natural_bonus"),
            "forced": ("forced_bonus", "forced_wincap", "forced_contract"),
        }
        self.bet_modes = []
        self._last_shift_initialized = True
