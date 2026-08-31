"""Last Shift round and bonus state transitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Sequence

from games.last_shift.game_calculations import (
    Board,
    DepartureGroup,
    WinningGroup,
    apply_departure_modifier,
    apply_level_modifier,
    derive_contribution_bucket,
    derive_scatter_positions,
    evaluate_pay_anywhere,
    load_selected_columns,
    make_departure_groups,
    modifier_reason_for,
    select_cargo_columns,
    serialize_wins,
    validate_board,
)
from games.last_shift.contract_fixtures import CARGO_BOARD, LOSS_BOARD
from games.last_shift.game_config import GameConfig
from games.last_shift.game_events import EventLedger
from src.config.paths import PATH_TO_GAMES
from src.state.state import GeneralGameState


STAGES = ("yard", "mainline", "redline")
RESET_LEVELS = {"yard": 0, "mainline": 1, "redline": 2}


def stage_for_departures(departures: int) -> str:
    if departures >= 4:
        return "redline"
    if departures >= 2:
        return "mainline"
    return "yard"


def validate_trigger_positions(
    board: Board, positions: Sequence[int], config: GameConfig
) -> tuple[int, ...]:
    validate_board(board, config)
    normalized = tuple(positions)
    if len(normalized) < 4:
        raise ValueError("bonus trigger requires at least four positions")
    if len(set(normalized)) != len(normalized):
        raise ValueError("bonus trigger positions must be unique")
    if any(not isinstance(position, int) or position < 0 or position >= 30 for position in normalized):
        raise ValueError("bonus trigger position is outside the board")
    authoritative = derive_scatter_positions(board, config)
    if tuple(sorted(normalized)) != authoritative:
        raise ValueError("bonus trigger positions must point to S and equal the complete sorted scatter set")
    return normalized


@dataclass(frozen=True)
class RoundState:
    mode: str = "basegame"
    spin_index: int = 0
    cascade_index: int = 0
    column_levels: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    departures: int = 0
    stage: str = "yard"
    free_spins_remaining: int = 0
    round_payout_units: int = 0
    capped: bool = False

    def __post_init__(self):
        if self.mode not in ("basegame", "freegame"):
            raise ValueError("mode must be basegame or freegame")
        if len(self.column_levels) != 6 or any(
            level not in (0, 1, 2) for level in self.column_levels
        ):
            raise ValueError("stable levels must contain six values in range 0..2")
        if self.stage not in STAGES:
            raise ValueError("unknown bonus stage")
        if min(
            self.spin_index,
            self.cascade_index,
            self.departures,
            self.free_spins_remaining,
            self.round_payout_units,
        ) < 0:
            raise ValueError("round state counters cannot be negative")
        if self.mode == "basegame" and (
            self.departures != 0 or self.stage != "yard" or self.free_spins_remaining != 0
        ):
            raise ValueError("bonus-only state cannot leak into basegame")


class LastShiftStateMachine:
    """Deterministic state facade that can be called by an SDK GameState."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random()
        self.active_outcome_path = "natural"

    def reset_seed(self, sim: int, simulation_seed: int | None = None) -> int:
        """Match SDK simulation seeding while keeping RNG local to this game."""
        seed = (simulation_seed if simulation_seed is not None else sim) + 1
        self.rng.seed(seed)
        return seed

    def deterministic_trace(self, length: int = 8) -> tuple[int, ...]:
        """Small reproducibility trace; this is not a tuned reel draw."""
        if length < 1:
            raise ValueError("trace length must be positive")
        return tuple(self.rng.randrange(2**31) for _ in range(length))

    def new_base_round(self) -> tuple[RoundState, EventLedger]:
        self.active_outcome_path = "natural"
        state = RoundState()
        ledger = EventLedger(
            self.config.wincap_units,
            payout_quantum_units=self.config.payout_quantum_units,
        )
        ledger.round_start(
            mode=state.mode,
            levels=state.column_levels,
            departures=state.departures,
            stage=state.stage,
            free_spins_remaining=state.free_spins_remaining,
        )
        return state, ledger

    def trigger_natural_bonus(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
    ) -> RoundState:
        if state.mode != self.config.basegame_type:
            raise ValueError("natural bonus can only trigger from basegame")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        bonus_state = RoundState(
            mode=self.config.freegame_type,
            column_levels=state.column_levels,
            departures=0,
            stage="yard",
            free_spins_remaining=self.config.initial_free_spins,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )
        ledger.bonus_trigger(
            positions,
            self.config.initial_free_spins,
            bonus_state.column_levels,
            outcome_path="natural",
            board=board,
        )
        self.active_outcome_path = "natural"
        return bonus_state

    def build_forced_bonus(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
        criterion: str,
    ) -> RoundState:
        if criterion not in self.config.outcome_paths["forced"]:
            raise ValueError("forced outcome requires a dedicated forced criterion")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        bonus_state = RoundState(
            mode=self.config.freegame_type,
            column_levels=state.column_levels,
            departures=0,
            stage="yard",
            free_spins_remaining=self.config.initial_free_spins,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )
        ledger.bonus_trigger(
            positions,
            self.config.initial_free_spins,
            bonus_state.column_levels,
            outcome_path="forced",
            board=board,
        )
        self.active_outcome_path = "forced"
        return bonus_state

    def start_free_spin(self, state: RoundState, ledger: EventLedger) -> RoundState:
        if state.mode != self.config.freegame_type or state.free_spins_remaining <= 0:
            raise ValueError("no free spin is available")
        next_state = replace(
            state,
            spin_index=state.spin_index + 1,
            cascade_index=0,
            free_spins_remaining=state.free_spins_remaining - 1,
        )
        ledger.free_spin_start(
            next_state.spin_index,
            next_state.free_spins_remaining,
            next_state.column_levels,
            next_state.stage,
            next_state.departures,
        )
        return next_state

    def retrigger(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
    ) -> RoundState:
        if state.mode != self.config.freegame_type:
            raise ValueError("retrigger can only occur in freegame")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        next_state = replace(
            state,
            free_spins_remaining=(
                state.free_spins_remaining + self.config.retrigger_free_spins
            ),
        )
        ledger.retrigger(
            positions,
            self.config.retrigger_free_spins,
            next_state.free_spins_remaining,
            next_state.column_levels,
            next_state.stage,
            next_state.departures,
            board,
        )
        return next_state

    def resolve_departures(
        self,
        state: RoundState,
        ledger: EventLedger,
        groups: Sequence[DepartureGroup],
    ) -> RoundState:
        levels = list(state.column_levels)
        departures = state.departures
        selection_stage = (
            "yard"
            if state.mode == self.config.basegame_type
            else stage_for_departures(departures)
        )
        if any(group.stage_at_selection != selection_stage for group in groups):
            raise ValueError("departure stageAtSelection does not match frozen evaluation state")
        for group in groups:
            reset_level = 0 if state.mode == self.config.basegame_type else RESET_LEVELS[group.stage_at_selection]
            for column in group.columns:
                levels[column] = reset_level
            departures += len(group.columns)
            stage_after = (
                "yard"
                if state.mode == self.config.basegame_type
                else stage_for_departures(departures)
            )
            ledger.departure_resolve(
                group,
                [reset_level] * len(group.columns),
                departures if state.mode == self.config.freegame_type else 0,
                stage_after,
            )
        return replace(
            state,
            column_levels=tuple(levels),
            departures=departures if state.mode == self.config.freegame_type else 0,
            stage=(
                stage_for_departures(departures)
                if state.mode == self.config.freegame_type
                else "yard"
            ),
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )

    def complete_round(self, state: RoundState, ledger: EventLedger) -> RoundState:
        if state.capped != ledger.capped:
            raise ValueError("state and ledger cap status do not match")
        final_levels = (0, 0, 0, 0, 0, 0)
        ledger.round_complete(final_levels)
        self.active_outcome_path = "natural"
        return RoundState(
            mode=self.config.basegame_type,
            column_levels=final_levels,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )

    def complete_bonus(
        self, state: RoundState, ledger: EventLedger, feature_payout_units: int
    ) -> RoundState:
        if state.mode != self.config.freegame_type:
            raise ValueError("bonus completion requires freegame state")
        if state.capped != ledger.capped:
            raise ValueError("state and ledger cap status do not match")
        if ledger.capped:
            return self.complete_round(state, ledger)
        if state.free_spins_remaining != 0:
            raise ValueError("bonus completion requires zero remaining spins")
        if feature_payout_units != ledger.feature_payout_units:
            raise ValueError("bonus payout must be ledger-derived")
        ledger.bonus_complete(ledger.feature_payout_units)
        return self.complete_round(state, ledger)


class GameState(GeneralGameState):
    """Stake SDK adapter for the two deterministic contract-proof books."""

    _SUPPORTED_CRITERIA = frozenset({"contract_loss", "contract_departure"})

    def assign_special_sym_function(self) -> None:
        self.special_symbol_functions = {}

    def run_freespin(self) -> None:
        raise RuntimeError("contract_proof does not cover freegame integration")

    def run_spin(self, sim: int, simulation_seed=None) -> None:
        self._assert_contract_output_isolated()
        if self.criteria not in self._SUPPORTED_CRITERIA:
            raise ValueError(f"unsupported contract_proof criterion: {self.criteria}")

        self.reset_seed(sim, simulation_seed)
        self.reset_book()
        ledger, final_state = self._produce_contract_book(self.criteria)

        self.book.events = deepcopy(ledger.events)
        final_units = ledger.events[-1]["finalPayoutUnits"]
        final_multiplier = final_units / self.config.payout_scale
        self.win_manager.update_spinwin(final_multiplier)
        self.win_manager.update_gametype_wins(self.gametype)
        self.update_final_win()
        self._last_shift_state = final_state
        self.imprint_wins()

    def _assert_contract_output_isolated(self) -> None:
        canonical_library = (
            Path(PATH_TO_GAMES) / "last_shift" / "library"
        ).resolve()
        for target_path in self._cached_output_paths():
            if (
                target_path == canonical_library
                or canonical_library in target_path.parents
            ):
                raise RuntimeError(
                    "contract_proof cannot write to the canonical Last Shift "
                    f"library: {target_path}"
                )

    def _cached_output_paths(self):
        values = list(vars(self.output_files).values())
        visited_containers = set()
        while values:
            value = values.pop()
            if isinstance(value, (str, Path)):
                path = Path(value)
                if path.is_absolute():
                    yield path.resolve()
                continue

            if isinstance(value, dict):
                container_id = id(value)
                if container_id in visited_containers:
                    continue
                visited_containers.add(container_id)
                values.extend(value.keys())
                values.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                container_id = id(value)
                if container_id in visited_containers:
                    continue
                visited_containers.add(container_id)
                values.extend(value)

    def _produce_contract_book(self, criterion: str) -> tuple[EventLedger, RoundState]:
        state = RoundState()
        ledger = EventLedger(
            self.config.wincap_units,
            payout_quantum_units=self.config.payout_quantum_units,
        )
        ledger.round_start(
            mode=state.mode,
            levels=state.column_levels,
            departures=state.departures,
            stage=state.stage,
            free_spins_remaining=state.free_spins_remaining,
        )

        if criterion == "contract_loss":
            ledger.board_reveal(LOSS_BOARD)
            ledger.no_win()
        elif criterion == "contract_departure":
            state = self._produce_departure(state, ledger)
        else:  # Defensive even though run_spin rejects before resetting a book.
            raise ValueError(f"unsupported contract_proof criterion: {criterion}")

        final_levels = (0, 0, 0, 0, 0, 0)
        ledger.round_complete(final_levels)
        return ledger, RoundState(
            column_levels=final_levels,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )

    def _produce_departure(self, state: RoundState, ledger: EventLedger) -> RoundState:
        board = CARGO_BOARD
        ledger.board_reveal(board)
        modifier_reason = None

        for load_level in (1, 2, 3):
            evaluated, payout_units = evaluate_pay_anywhere(board, self.config)
            ledger.win_result(
                payout_units,
                groups=serialize_wins(evaluated),
                contribution_bucket=derive_contribution_bucket(
                    state.mode, modifier_reason, ()
                ),
            )
            winning_positions = sorted(
                {position for win in evaluated for position in win.positions}
            )
            ledger.symbols_remove(winning_positions)
            selections = select_cargo_columns(
                board,
                tuple(WinningGroup(win.symbol, win.positions) for win in evaluated),
                self.config,
            )
            next_levels, transitions = load_selected_columns(
                state.column_levels, selections
            )
            ledger.columns_load(transitions)
            state = replace(state, column_levels=next_levels)

            if load_level < 3:
                ledger.cascade_refill(CARGO_BOARD)
                board, changes = self._apply_level_modifiers(
                    CARGO_BOARD, transitions
                )
                modifier_reason = modifier_reason_for(transitions)
                ledger.board_modifier(modifier_reason, changes, board)
                continue

            completed_columns = tuple(
                transition.column
                for transition in transitions
                if transition.level_after == 3
            )
            groups = make_departure_groups(
                completed_columns,
                state.stage,
                {2: 2, 3: 3},
                departure_sequence=1,
                mode=state.mode,
            )
            for group in groups:
                ledger.departure_prepare(group)
            ledger.cascade_refill(CARGO_BOARD)
            board, changes = apply_departure_modifier(
                CARGO_BOARD, groups, self.config
            )
            ledger.board_modifier("departure", changes, board)
            evaluated, payout_units = evaluate_pay_anywhere(
                board, self.config, groups
            )
            ledger.win_result(
                payout_units,
                groups=serialize_wins(evaluated),
                applied_departure_ids=tuple(
                    group.departure_id for group in groups
                ),
                contribution_bucket=derive_contribution_bucket(
                    state.mode, "departure", groups
                ),
            )
            state = LastShiftStateMachine(self.config).resolve_departures(
                state, ledger, groups
            )

        return state

    def _apply_level_modifiers(self, board, transitions):
        resulting_board = board
        changes = []
        for transition in transitions:
            resulting_board, transition_changes = apply_level_modifier(
                resulting_board, transition, self.config
            )
            changes.extend(transition_changes)
        return resulting_board, tuple(changes)
