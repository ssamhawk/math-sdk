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


class EventLedger:
    """Append-only canonical event ledger for one preselected game outcome."""

    def __init__(self, max_win_units: int, starting_payout_units: int = 0):
        if max_win_units <= 0:
            raise ValueError("max win must be positive")
        if starting_payout_units < 0 or starting_payout_units > max_win_units:
            raise ValueError("starting payout is outside the cap")
        self.max_win_units = max_win_units
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
        return self._append(
            "bonus_trigger",
            positions=list(positions),
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
        self, positions: Sequence[int], added_spins: int, free_spins_after: int
    ) -> dict[str, Any]:
        return self._append(
            "retrigger",
            positions=list(positions),
            addedSpins=added_spins,
            freeSpinsAfter=free_spins_after,
        )

    def free_spin_complete(
        self, payout_units: int, spins_remaining: int, levels: Sequence[int]
    ) -> dict[str, Any]:
        return self._append(
            "free_spin_complete",
            payoutUnits=payout_units,
            spinsRemaining=spins_remaining,
            levels=list(levels),
        )

    def bonus_complete(self, feature_payout_units: int) -> dict[str, Any]:
        return self._append(
            "bonus_complete",
            featurePayoutUnits=feature_payout_units,
            roundPayoutUnits=self.round_payout_units,
        )

    def round_complete(self, final_levels: Sequence[int]) -> dict[str, Any]:
        event = self._append(
            "round_complete",
            allow_after_cap=True,
            finalPayoutUnits=self.round_payout_units,
            finalLevels=list(final_levels),
            capped=self.capped,
            priorPayoutUnits=self.prior_payout_units,
            contributionUnits=dict(self.contribution_units),
        )
        self.closed = True
        return event


def event_types(events: Sequence[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def validate_contract(events: Sequence[dict[str, Any]], max_win_units: int) -> None:
    """Validate the Last Shift contract beyond the SDK's generic RGS checks."""
    if not events:
        raise ValueError("event stream is empty")
    payout_after = 0
    prior_payout_units = 0
    saw_payout = False
    contribution_units = {bucket: 0 for bucket in CONTRIBUTION_BUCKETS}
    max_win_index = None

    for index, event in enumerate(events):
        if event.get("index") != index:
            raise ValueError("event indices are not contiguous")
        event_type = event.get("type")
        previous_type = events[index - 1].get("type") if index else None
        if max_win_index is not None and event_type != "round_complete":
            raise ValueError("gameplay event found after max_win")
        if event_type == "cascade_refill":
            if previous_type not in ("columns_load", "departure_prepare"):
                raise ValueError("cascade_refill is outside canonical order")
            board = event.get("resultingBoard")
            if not isinstance(board, list) or len(board) != 6 or any(
                not isinstance(column, list) or len(column) != 5 for column in board
            ):
                raise ValueError("cascade_refill requires a deep authoritative resultingBoard")
        elif event_type == "win_result":
            if previous_type not in ("board_reveal", "board_modifier"):
                raise ValueError("win_result is outside canonical order")
            requested = event.get("requestedStepPayoutUnits")
            applied = event.get("stepPayoutUnits")
            before = event.get("roundPayoutBeforeUnits")
            cumulative = event.get("roundPayoutAfterUnits")
            bucket = event.get("contributionBucket")
            if not all(
                isinstance(value, int) for value in (requested, applied, before, cumulative)
            ):
                raise ValueError("canonical payout values must be integers")
            if not saw_payout:
                prior_payout_units = before
                payout_after = before
                saw_payout = True
            if before != payout_after:
                raise ValueError("payout before value does not reconcile")
            if requested < 0 or applied < 0 or applied > requested:
                raise ValueError("invalid payout delta")
            expected = min(payout_after + requested, max_win_units)
            if cumulative != expected or applied != expected - payout_after:
                raise ValueError("payout delta does not reconcile under cap")
            if bucket not in contribution_units:
                raise ValueError("unknown contribution bucket")
            contribution_units[bucket] += applied
            payout_after = cumulative
        elif event_type == "max_win":
            if event.get("amountUnits") != max_win_units or payout_after != max_win_units:
                raise ValueError("max_win does not reconcile to the cap")
            if index == 0 or events[index - 1].get("type") != "win_result":
                raise ValueError("max_win must immediately follow the capped payout delta")
            max_win_index = index
        elif event_type == "symbols_remove":
            if previous_type != "win_result":
                raise ValueError("symbols_remove must follow win_result")
        elif event_type == "columns_load":
            if previous_type != "symbols_remove":
                raise ValueError("columns_load must follow symbols_remove")
        elif event_type == "departure_prepare":
            if previous_type not in ("columns_load", "departure_prepare"):
                raise ValueError("departure_prepare is outside canonical order")
        elif event_type == "board_modifier":
            if previous_type not in ("board_reveal", "cascade_refill"):
                raise ValueError("board_modifier is outside canonical order")
        elif event_type == "departure_resolve":
            if previous_type != "win_result":
                raise ValueError("departure_resolve must follow win_result")
        elif event_type == "round_complete":
            if index != len(events) - 1:
                raise ValueError("round_complete must be terminal")
            prior = event.get("priorPayoutUnits", 0)
            if not saw_payout:
                prior_payout_units = prior
                payout_after = prior
            if prior != prior_payout_units or event.get("finalPayoutUnits") != payout_after:
                raise ValueError("round_complete payout does not reconcile")
            if event.get("contributionUnits") != contribution_units:
                raise ValueError("mechanic contribution ledger is not additive")

    if payout_after > max_win_units:
        raise ValueError("event stream exceeds max win")
