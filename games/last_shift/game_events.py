"""Canonical Last Shift event builders with integer payout reconciliation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Iterable, Sequence

from games.last_shift.game_calculations import (
    DepartureGroup,
    LoadTransition,
    SymbolChange,
)


class TerminalPayoutError(RuntimeError):
    """Raised when payout generation is attempted after max_win."""


CONTRIBUTION_BUCKETS = (
    "base_plain",
    "base_modifier",
    "base_single_departure",
    "base_coupled_departure",
    "bonus_yard",
    "bonus_mainline",
    "bonus_redline",
)


def _validate_bonus_positions(positions: Sequence[int]) -> list[int]:
    normalized = list(positions)
    if len(normalized) < 4:
        raise ValueError("bonus positions require at least four entries")
    if len(normalized) != len(set(normalized)):
        raise ValueError("bonus positions must be unique")
    if any(not isinstance(position, int) or position < 0 or position >= 30 for position in normalized):
        raise ValueError("bonus position is outside the board")
    return normalized


class EventLedger:
    """Append-only canonical event ledger for one preselected game outcome."""

    def __init__(
        self,
        max_win_units: int,
        starting_payout_units: int = 0,
        payout_quantum_units: int = 10,
    ):
        if max_win_units <= 0:
            raise ValueError("max win must be positive")
        if starting_payout_units < 0 or starting_payout_units > max_win_units:
            raise ValueError("starting payout is outside the cap")
        if payout_quantum_units <= 0:
            raise ValueError("payout quantum must be positive")
        if starting_payout_units and (
            starting_payout_units < payout_quantum_units
            or starting_payout_units % payout_quantum_units
        ):
            raise ValueError("starting payout violates the SDK payout quantum")
        self.max_win_units = max_win_units
        self.payout_quantum_units = payout_quantum_units
        self.round_payout_units = starting_payout_units
        self.capped = starting_payout_units == max_win_units
        self.closed = False
        self.contribution_units = {bucket: 0 for bucket in CONTRIBUTION_BUCKETS}
        self.prior_payout_units = starting_payout_units
        self.events: list[dict[str, Any]] = []

    def _append(
        self, event_type: str, *, allow_after_cap: bool = False, **payload: Any
    ) -> dict[str, Any]:
        if self.closed:
            raise TerminalPayoutError("round_complete is terminal")
        if self.capped and not allow_after_cap:
            raise TerminalPayoutError("no gameplay event is allowed after max_win")
        event = {"index": len(self.events), "type": event_type, **payload}
        self.events.append(event)
        return event

    def board_reveal(self, board: Sequence[Sequence[str]]) -> dict[str, Any]:
        return self._append("board_reveal", board=[list(column) for column in board])

    def round_start(
        self,
        mode: str,
        levels: Sequence[int],
        departures: int,
        stage: str,
        free_spins_remaining: int,
    ) -> dict[str, Any]:
        if self.events:
            raise ValueError("round_start must be the first event")
        return self._append(
            "round_start",
            mode=mode,
            levels=list(levels),
            departures=departures,
            stage=stage,
            freeSpinsRemaining=free_spins_remaining,
            roundPayoutUnits=self.round_payout_units,
        )

    def cascade_refill(self, board: Sequence[Sequence[str]]) -> dict[str, Any]:
        return self._append(
            "cascade_refill", resultingBoard=deepcopy([list(column) for column in board])
        )

    def board_modifier(
        self,
        reason: str,
        changes: Sequence[SymbolChange],
        resulting_board: Sequence[Sequence[str]],
    ) -> dict[str, Any]:
        return self._append(
            "board_modifier",
            reason=reason,
            changes=[asdict(change) for change in changes],
            resultingBoard=[list(column) for column in resulting_board],
        )

    def win_result(
        self,
        requested_step_payout_units: int,
        groups: Sequence[dict[str, Any]] = (),
        applied_departure_ids: Sequence[str] = (),
        contribution_bucket: str = "base_plain",
    ) -> dict[str, Any]:
        if self.capped:
            raise TerminalPayoutError("no payout event is allowed after max_win")
        if not isinstance(requested_step_payout_units, int) or requested_step_payout_units < 0:
            raise ValueError("payout must be a non-negative integer unit amount")
        if requested_step_payout_units and (
            requested_step_payout_units < self.payout_quantum_units
            or requested_step_payout_units % self.payout_quantum_units
        ):
            raise ValueError("payout violates the SDK payout quantum")
        if contribution_bucket not in CONTRIBUTION_BUCKETS:
            raise ValueError(f"unknown contribution bucket: {contribution_bucket}")
        payout_before = self.round_payout_units
        available = self.max_win_units - self.round_payout_units
        applied = min(requested_step_payout_units, available)
        self.round_payout_units += applied
        self.contribution_units[contribution_bucket] += applied
        event = self._append(
            "win_result",
            groups=list(groups),
            requestedStepPayoutUnits=requested_step_payout_units,
            stepPayoutUnits=applied,
            roundPayoutBeforeUnits=payout_before,
            roundPayoutAfterUnits=self.round_payout_units,
            appliedDepartureIds=list(applied_departure_ids),
            contributionBucket=contribution_bucket,
        )
        if self.round_payout_units == self.max_win_units:
            self.capped = True
            self._append(
                "max_win", allow_after_cap=True, amountUnits=self.max_win_units
            )
        return event

    def no_win(self) -> dict[str, Any]:
        return self._append("no_win", roundPayoutUnits=self.round_payout_units)

    def symbols_remove(self, positions: Iterable[int]) -> dict[str, Any]:
        return self._append("symbols_remove", positions=sorted(set(positions)))

    def columns_load(self, transitions: Sequence[LoadTransition]) -> dict[str, Any]:
        return self._append(
            "columns_load",
            transitions=[
                {
                    "column": transition.column,
                    "levelBefore": transition.level_before,
                    "levelAfter": transition.level_after,
                    "sourcePositions": list(transition.source_positions),
                    "regularWinningCount": transition.regular_winning_count,
                    "selectionRank": transition.selection_rank,
                }
                for transition in transitions
            ],
        )

    def departure_prepare(self, group: DepartureGroup) -> dict[str, Any]:
        return self._append(
            "departure_prepare",
            departureId=group.departure_id,
            kind=group.kind,
            columns=list(group.columns),
            components=list(group.components),
            multiplier=group.multiplier,
            stageAtSelection=group.stage_at_selection,
        )

    def departure_resolve(
        self,
        group: DepartureGroup,
        reset_levels: Sequence[int],
        departures_after: int,
        stage_after: str,
    ) -> dict[str, Any]:
        return self._append(
            "departure_resolve",
            departureId=group.departure_id,
            columns=list(group.columns),
            resetLevels=list(reset_levels),
            departuresAfter=departures_after,
            stageAfter=stage_after,
        )

    def bonus_trigger(
        self,
        positions: Sequence[int],
        awarded_spins: int,
        levels: Sequence[int],
        outcome_path: str = "natural",
    ) -> dict[str, Any]:
        valid_positions = _validate_bonus_positions(positions)
        return self._append(
            "bonus_trigger",
            positions=valid_positions,
            awardedSpins=awarded_spins,
            startingLevels=list(levels),
            outcomePath=outcome_path,
        )

    def free_spin_start(
        self,
        spin_index: int,
        spins_remaining: int,
        levels: Sequence[int],
        stage: str,
        departures: int,
    ) -> dict[str, Any]:
        return self._append(
            "free_spin_start",
            spinIndex=spin_index,
            spinsRemaining=spins_remaining,
            levels=list(levels),
            stage=stage,
            departures=departures,
        )

    def retrigger(
        self,
        positions: Sequence[int],
        added_spins: int,
        free_spins_after: int,
        levels: Sequence[int],
        stage: str,
        departures: int,
    ) -> dict[str, Any]:
        valid_positions = _validate_bonus_positions(positions)
        return self._append(
            "retrigger",
            positions=valid_positions,
            addedSpins=added_spins,
            freeSpinsAfter=free_spins_after,
            levels=list(levels),
            stage=stage,
            departures=departures,
            roundPayoutUnits=self.round_payout_units,
        )

    def free_spin_complete(
        self, payout_units: int, spins_remaining: int, levels: Sequence[int]
    ) -> dict[str, Any]:
        if not isinstance(payout_units, int) or payout_units < 0 or (
            payout_units and payout_units % self.payout_quantum_units
        ):
            raise ValueError("free-spin payout violates the SDK payout quantum")
        return self._append(
            "free_spin_complete",
            payoutUnits=payout_units,
            spinsRemaining=spins_remaining,
            levels=list(levels),
            roundPayoutUnits=self.round_payout_units,
        )

    def bonus_complete(self, feature_payout_units: int) -> dict[str, Any]:
        if not isinstance(feature_payout_units, int) or feature_payout_units < 0 or (
            feature_payout_units and feature_payout_units % self.payout_quantum_units
        ):
            raise ValueError("bonus payout violates the SDK payout quantum")
        return self._append(
            "bonus_complete",
            featurePayoutUnits=feature_payout_units,
            roundPayoutUnits=self.round_payout_units,
        )

    def round_complete(
        self,
        final_levels: Sequence[int],
        final_mode: str = "basegame",
        final_stage: str = "yard",
        final_departures: int = 0,
        final_free_spins_remaining: int = 0,
    ) -> dict[str, Any]:
        event = self._append(
            "round_complete",
            allow_after_cap=True,
            finalPayoutUnits=self.round_payout_units,
            finalLevels=list(final_levels),
            finalMode=final_mode,
            finalStage=final_stage,
            finalDepartures=final_departures,
            finalFreeSpinsRemaining=final_free_spins_remaining,
            capped=self.capped,
            priorPayoutUnits=self.prior_payout_units,
            contributionUnits=dict(self.contribution_units),
        )
        self.closed = True
        return event


def event_types(events: Sequence[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def validate_contract(events: Sequence[dict[str, Any]], max_win_units: int) -> None:
    """Strictly validate a complete natural-base outcome event stream."""
    event_whitelist = {
        "round_start",
        "board_reveal",
        "board_modifier",
        "win_result",
        "no_win",
        "symbols_remove",
        "columns_load",
        "departure_prepare",
        "cascade_refill",
        "departure_resolve",
        "bonus_trigger",
        "free_spin_start",
        "retrigger",
        "free_spin_complete",
        "bonus_complete",
        "max_win",
        "round_complete",
    }
    regular_symbols = set("ABCDEFGH")
    valid_symbols = regular_symbols | {"W", "S"}
    stages = ("yard", "mainline", "redline")
    reset_levels = {"yard": 0, "mainline": 1, "redline": 2}
    payout_quantum = 10

    def stage_for(departures: int) -> str:
        if departures >= 4:
            return "redline"
        if departures >= 2:
            return "mainline"
        return "yard"

    def check_levels(value: Any, label: str) -> list[int]:
        if not isinstance(value, list) or len(value) != 6 or any(
            not isinstance(level, int) or level not in (0, 1, 2) for level in value
        ):
            raise ValueError(f"{label} must be six stable levels in range 0..2")
        return list(value)

    def check_board(value: Any, label: str) -> list[list[str]]:
        if not isinstance(value, list) or len(value) != 6 or any(
            not isinstance(column, list) or len(column) != 5 for column in value
        ):
            raise ValueError(f"{label} must contain a 6x5 board")
        if any(symbol not in valid_symbols for column in value for symbol in column):
            raise ValueError(f"{label} contains an unknown symbol")
        return deepcopy(value)

    def check_positions(value: Any, label: str, minimum: int = 0) -> list[int]:
        if not isinstance(value, list) or len(value) < minimum or any(
            not isinstance(position, int) or position < 0 or position >= 30
            for position in value
        ):
            raise ValueError(f"{label} contains an invalid board position")
        if len(value) != len(set(value)):
            raise ValueError(f"{label} contains duplicate positions")
        return list(value)

    if not events:
        raise ValueError("event stream is empty")
    if events[0].get("type") != "round_start":
        raise ValueError("round_start must be first")
    if events[-1].get("type") != "round_complete":
        raise ValueError("terminal round_complete is required")

    expected = {"round_start"}
    mode = "basegame"
    levels = [0] * 6
    departures = 0
    stage = "yard"
    free_spins = 0
    spin_index = 0
    payout_after = 0
    contribution_units = {bucket: 0 for bucket in CONTRIBUTION_BUCKETS}
    current_board: list[list[str]] | None = None
    regular_win_positions: set[int] = set()
    winning_positions: set[int] = set()
    scheduled_modifiers: dict[int, int] = {}
    full_columns: set[int] = set()
    prepared: dict[str, dict[str, Any]] = {}
    unresolved_departures: set[str] = set()
    departure_evaluation = False
    spin_start_payout: int | None = None
    bonus_start_payout: int | None = None
    saw_max_win = False

    for index, event in enumerate(events):
        if event.get("index") != index:
            raise ValueError("event indices are not contiguous")
        event_type = event.get("type")
        if event_type not in event_whitelist:
            raise ValueError(f"unknown event type: {event_type}")
        if event_type not in expected:
            raise ValueError(f"illegal {event_type} after {events[index - 1].get('type') if index else 'start'}")

        if event_type == "round_start":
            if event.get("mode") != "basegame":
                raise ValueError("complete round must start in basegame")
            levels = check_levels(event.get("levels"), "start levels")
            if levels != [0] * 6:
                raise ValueError("base round must start with zero levels")
            if event.get("departures") != 0 or event.get("stage") != "yard":
                raise ValueError("base round has invalid starting stage state")
            if event.get("freeSpinsRemaining") != 0 or event.get("roundPayoutUnits") != 0:
                raise ValueError("base round has invalid starting counters")
            expected = {"board_reveal"}

        elif event_type == "board_reveal":
            current_board = check_board(event.get("board"), "board_reveal.board")
            expected = {"board_modifier", "win_result", "no_win"}

        elif event_type == "cascade_refill":
            if full_columns:
                raise ValueError("full transitions are missing departure_prepare")
            current_board = check_board(
                event.get("resultingBoard"), "cascade_refill.resultingBoard"
            )
            departure_evaluation = bool(unresolved_departures)
            if departure_evaluation or scheduled_modifiers:
                expected = {"board_modifier"}
            else:
                expected = {"board_modifier", "win_result", "no_win"}

        elif event_type == "board_modifier":
            current_board = check_board(
                event.get("resultingBoard"), "board_modifier.resultingBoard"
            )
            changes = event.get("changes")
            if not isinstance(changes, list):
                raise ValueError("board_modifier changes must be a list")
            if departure_evaluation:
                if event.get("reason") != "departure":
                    raise ValueError("departure evaluation requires departure modifier")
                departing_columns = {
                    column
                    for departure_id in unresolved_departures
                    for column in prepared[departure_id]["columns"]
                }
                changed_positions = {change.get("position") for change in changes}
                required_positions = {
                    row * 6 + column for column in departing_columns for row in range(5)
                }
                if changed_positions != required_positions:
                    raise ValueError("departure modifier must cover every departing position")
                if any(current_board[column] != ["W"] * 5 for column in departing_columns):
                    raise ValueError("departure resulting board must contain full wild columns")
            elif scheduled_modifiers:
                counts: dict[int, int] = {}
                for change in changes:
                    column = change.get("column")
                    position = change.get("position")
                    if column not in scheduled_modifiers or not isinstance(position, int) or position % 6 != column:
                        raise ValueError("load modifier change does not match scheduled column")
                    counts[column] = counts.get(column, 0) + 1
                if counts != scheduled_modifiers:
                    raise ValueError("load modifier change count does not match entered level")
                scheduled_modifiers = {}
            expected = {"win_result"} if departure_evaluation else {"win_result", "no_win"}

        elif event_type == "win_result":
            if current_board is None:
                raise ValueError("win_result has no authoritative board")
            requested = event.get("requestedStepPayoutUnits")
            applied = event.get("stepPayoutUnits")
            before = event.get("roundPayoutBeforeUnits")
            cumulative = event.get("roundPayoutAfterUnits")
            if not all(isinstance(value, int) for value in (requested, applied, before, cumulative)):
                raise ValueError("canonical payout values must be integers")
            if requested < payout_quantum or requested % payout_quantum:
                raise ValueError("win_result violates the SDK payout quantum")
            if before != payout_after:
                raise ValueError("payout before value does not reconcile")
            expected_payout = min(before + requested, max_win_units)
            if applied != expected_payout - before or cumulative != expected_payout:
                raise ValueError("payout delta does not reconcile under cap")
            if applied and (applied < payout_quantum or applied % payout_quantum):
                raise ValueError("applied payout violates the SDK payout quantum")
            bucket = event.get("contributionBucket")
            if bucket not in contribution_units:
                raise ValueError("unknown contribution bucket")
            contribution_units[bucket] += applied
            payout_after = cumulative

            groups = event.get("groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError("win_result requires authoritative winning groups")
            regular_win_positions = set()
            winning_positions = set()
            for group in groups:
                symbol = group.get("symbol")
                positions = check_positions(group.get("positions"), "winning group positions", 1)
                if symbol not in regular_symbols:
                    raise ValueError("winning group symbol must be regular")
                for position in positions:
                    column, row = position % 6, position // 6
                    landed = current_board[column][row]
                    if landed not in (symbol, "W"):
                        raise ValueError("winning group position does not match board")
                    if landed == symbol:
                        regular_win_positions.add(position)
                winning_positions.update(positions)

            applied_ids = event.get("appliedDepartureIds")
            if not isinstance(applied_ids, list) or len(applied_ids) != len(set(applied_ids)):
                raise ValueError("applied departure IDs must be a unique list")
            if departure_evaluation:
                if set(applied_ids) != unresolved_departures:
                    raise ValueError("departure payout IDs do not match prepared departures")
            elif applied_ids:
                raise ValueError("normal payout cannot reference a departure")

            if payout_after == max_win_units:
                expected = {"max_win"}
            elif departure_evaluation:
                expected = {"departure_resolve"}
            else:
                expected = {"symbols_remove"}

        elif event_type == "no_win":
            if event.get("roundPayoutUnits") != payout_after:
                raise ValueError("no_win payout does not reconcile")
            if departure_evaluation:
                raise ValueError("departure evaluation cannot continue after no_win")
            expected = (
                {"bonus_trigger", "round_complete"}
                if mode == "basegame"
                else {"retrigger", "free_spin_complete"}
            )

        elif event_type == "symbols_remove":
            positions = set(check_positions(event.get("positions"), "symbols_remove positions", 1))
            if positions != winning_positions:
                raise ValueError("symbols_remove must equal the winning-position union")
            expected = {"columns_load"}

        elif event_type == "columns_load":
            transitions = event.get("transitions")
            if not isinstance(transitions, list) or len(transitions) > 2:
                raise ValueError("columns_load must contain at most two transitions")
            columns = [transition.get("column") for transition in transitions]
            if len(columns) != len(set(columns)):
                raise ValueError("columns_load contains duplicate selected columns")
            source_union: set[int] = set()
            ranking = []
            scheduled_modifiers = {}
            full_columns = set()
            for transition in transitions:
                column = transition.get("column")
                before = transition.get("levelBefore")
                after = transition.get("levelAfter")
                count = transition.get("regularWinningCount")
                rank = transition.get("selectionRank")
                sources = check_positions(
                    transition.get("sourcePositions"), "cargo source positions", 2
                )
                if not isinstance(column, int) or column < 0 or column >= 6:
                    raise ValueError("load column is outside the board")
                if any(position % 6 != column for position in sources):
                    raise ValueError("cargo source belongs to another column")
                if source_union.intersection(sources):
                    raise ValueError("cargo source position is duplicated across transitions")
                source_union.update(sources)
                if count != len(sources) or count < 2:
                    raise ValueError("regularWinningCount does not match cargo sources")
                if not set(sources).issubset(regular_win_positions):
                    raise ValueError("cargo source is not a regular winning position")
                if before != levels[column] or after != min(before + 1, 3):
                    raise ValueError("load level transition does not match state")
                if after == 3:
                    full_columns.add(column)
                else:
                    levels[column] = after
                    scheduled_modifiers[column] = after
                ranking.append((count, column, rank))
            expected_ranking = [
                (count, column, rank)
                for rank, (count, column, _) in enumerate(
                    sorted(ranking, key=lambda value: (-value[0], value[1])), start=1
                )
            ]
            if sorted(ranking, key=lambda value: value[2]) != expected_ranking:
                raise ValueError("load selection rank does not match count/tie ordering")
            expected = {"departure_prepare"} if full_columns else {"cascade_refill"}

        elif event_type == "departure_prepare":
            departure_id = event.get("departureId")
            columns = event.get("columns")
            components = event.get("components")
            kind = event.get("kind")
            if not isinstance(departure_id, str) or not departure_id or departure_id in prepared:
                raise ValueError("departureId must be unique and non-empty")
            if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
                raise ValueError("departure columns must be a unique non-empty list")
            if kind == "single" and len(columns) != 1:
                raise ValueError("single departure must contain one column")
            if kind == "coupled" and (
                len(columns) != 2 or sorted(columns)[1] != sorted(columns)[0] + 1
            ):
                raise ValueError("coupled departure columns must be directly adjacent")
            if kind not in ("single", "coupled"):
                raise ValueError("unknown departure kind")
            if not set(columns).issubset(full_columns):
                raise ValueError("prepared departure has no matching full transition")
            if not isinstance(components, list) or len(components) != len(columns) or any(
                not isinstance(component, int) or component <= 0 for component in components
            ):
                raise ValueError("departure components are invalid")
            if event.get("multiplier") != sum(components):
                raise ValueError("departure multiplier must equal component sum")
            if event.get("stageAtSelection") != stage:
                raise ValueError("departure stageAtSelection does not match state")
            prepared[departure_id] = {
                "columns": list(columns),
                "stage": stage,
                "resolved": False,
            }
            unresolved_departures.add(departure_id)
            full_columns.difference_update(columns)
            expected = {"departure_prepare"} if full_columns else {"cascade_refill"}

        elif event_type == "departure_resolve":
            departure_id = event.get("departureId")
            if departure_id not in unresolved_departures:
                raise ValueError("departure_resolve ID was not prepared or was already resolved")
            prepared_group = prepared[departure_id]
            columns = prepared_group["columns"]
            if event.get("columns") != columns:
                raise ValueError("departure_resolve columns do not match prepare")
            reset = 0 if mode == "basegame" else reset_levels[prepared_group["stage"]]
            if event.get("resetLevels") != [reset] * len(columns):
                raise ValueError("departure reset levels do not match selection stage")
            next_departures = departures + len(columns) if mode == "freegame" else 0
            next_stage = stage_for(next_departures) if mode == "freegame" else "yard"
            if event.get("departuresAfter") != next_departures or event.get("stageAfter") != next_stage:
                raise ValueError("departure counters/stage do not reconcile")
            for column in columns:
                levels[column] = reset
            departures = next_departures
            stage = next_stage
            unresolved_departures.remove(departure_id)
            prepared_group["resolved"] = True
            if unresolved_departures:
                expected = {"departure_resolve"}
            else:
                departure_evaluation = False
                expected = {"free_spin_complete"} if mode == "freegame" else {"round_complete"}

        elif event_type == "bonus_trigger":
            if mode != "basegame":
                raise ValueError("bonus_trigger can only enter from basegame")
            check_positions(event.get("positions"), "bonus trigger positions", 4)
            if event.get("awardedSpins") != 10:
                raise ValueError("natural bonus must award ten spins")
            if event.get("startingLevels") != levels:
                raise ValueError("bonus starting levels do not match base state")
            if event.get("outcomePath") not in ("natural", "forced"):
                raise ValueError("bonus outcome path is invalid")
            mode = "freegame"
            departures = 0
            stage = "yard"
            free_spins = 10
            bonus_start_payout = payout_after
            expected = {"free_spin_start"}

        elif event_type == "free_spin_start":
            if mode != "freegame" or free_spins <= 0:
                raise ValueError("free_spin_start has no available bonus spin")
            spin_index += 1
            free_spins -= 1
            if (
                event.get("spinIndex") != spin_index
                or event.get("spinsRemaining") != free_spins
                or event.get("levels") != levels
                or event.get("stage") != stage
                or event.get("departures") != departures
            ):
                raise ValueError("free_spin_start state does not reconcile")
            spin_start_payout = payout_after
            expected = {"board_reveal"}

        elif event_type == "retrigger":
            if mode != "freegame":
                raise ValueError("retrigger requires freegame state")
            check_positions(event.get("positions"), "retrigger positions", 4)
            if event.get("addedSpins") != 4 or event.get("freeSpinsAfter") != free_spins + 4:
                raise ValueError("retrigger spin counters do not reconcile")
            if (
                event.get("levels") != levels
                or event.get("stage") != stage
                or event.get("departures") != departures
                or event.get("roundPayoutUnits") != payout_after
            ):
                raise ValueError("retrigger changed persistent bonus state")
            free_spins += 4
            expected = {"free_spin_complete"}

        elif event_type == "free_spin_complete":
            if mode != "freegame" or spin_start_payout is None:
                raise ValueError("free_spin_complete has no matching start")
            if (
                event.get("payoutUnits") != payout_after - spin_start_payout
                or event.get("roundPayoutUnits") != payout_after
                or event.get("spinsRemaining") != free_spins
                or event.get("levels") != levels
            ):
                raise ValueError("free_spin_complete payout/state does not reconcile")
            spin_start_payout = None
            expected = {"free_spin_start"} if free_spins else {"bonus_complete"}

        elif event_type == "bonus_complete":
            if mode != "freegame" or free_spins != 0 or bonus_start_payout is None:
                raise ValueError("bonus_complete has invalid feature state")
            if (
                event.get("featurePayoutUnits") != payout_after - bonus_start_payout
                or event.get("roundPayoutUnits") != payout_after
            ):
                raise ValueError("bonus_complete payout does not reconcile")
            expected = {"round_complete"}

        elif event_type == "max_win":
            if event.get("amountUnits") != max_win_units or payout_after != max_win_units:
                raise ValueError("max_win does not reconcile to cap")
            saw_max_win = True
            expected = {"round_complete"}

        elif event_type == "round_complete":
            if index != len(events) - 1:
                raise ValueError("round_complete must be terminal")
            final_levels = check_levels(event.get("finalLevels"), "final levels")
            if (
                event.get("finalMode") != "basegame"
                or final_levels != [0] * 6
                or event.get("finalStage") != "yard"
                or event.get("finalDepartures") != 0
                or event.get("finalFreeSpinsRemaining") != 0
            ):
                raise ValueError("round_complete final state is invalid")
            if event.get("priorPayoutUnits") != 0 or event.get("finalPayoutUnits") != payout_after:
                raise ValueError("round_complete payout does not reconcile")
            if event.get("contributionUnits") != contribution_units:
                raise ValueError("mechanic contribution ledger is not additive")
            if event.get("capped") is not saw_max_win:
                raise ValueError("round_complete capped flag does not match max_win")
            if saw_max_win and events[index - 1].get("type") != "max_win":
                raise ValueError("max_win must immediately precede capped final")
            if full_columns or unresolved_departures or any(
                not group["resolved"] for group in prepared.values()
            ):
                raise ValueError("departure lifecycle is incomplete at round_complete")
            expected = set()

    if expected:
        raise ValueError("event stream ended before terminal round_complete")
