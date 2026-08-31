"""Deterministic T4-C1 contract tests for Last Shift."""

from dataclasses import replace

import pytest

from games.last_shift.game_calculations import (
    CargoSelection,
    LoadTransition,
    WinningGroup,
    apply_departure_modifier,
    apply_level_modifier,
    flat_position,
    group_completed_columns,
    load_selected_columns,
    make_departure_groups,
    select_cargo_columns,
)
from games.last_shift.game_config import GameConfig
from games.last_shift.game_events import (
    EventLedger,
    TerminalPayoutError,
    event_types,
    validate_contract,
)
from games.last_shift.gamestate import LastShiftStateMachine, RoundState


@pytest.fixture
def config():
    return GameConfig()


@pytest.fixture
def machine(config):
    return LastShiftStateMachine(config)


def board_with_columns(*columns):
    assert len(columns) == 6
    return tuple(tuple(column) for column in columns)


def regular_board():
    return board_with_columns(
        "ABCDE",
        "ABCDE",
        "ABCDE",
        "ABCDE",
        "ABCDE",
        "ABCDE",
    )


def test_cargo_selection_uses_count_then_column_tie_break_and_rank(config):
    board = regular_board()
    group = WinningGroup(
        "A",
        tuple(flat_position(column, 0) for column in (0, 1, 2))
        + tuple(flat_position(column, row) for column in (0, 1) for row in (1, 2)),
    )
    # Rows 1 and 2 are B/C, so only the A positions are authoritative cargo.
    extra = WinningGroup(
        "B",
        tuple(flat_position(column, 1) for column in (0, 1, 2))
        + (flat_position(2, 2),),
    )
    third = WinningGroup("C", (flat_position(2, 2),))
    selections = select_cargo_columns(board, (group, extra, third), config)
    assert [(item.column, item.regular_winning_count, item.selection_rank) for item in selections] == [
        (2, 3, 1),
        (0, 2, 2),
    ]


def test_selection_loads_maximum_two_columns(config):
    board = regular_board()
    positions = tuple(
        flat_position(column, row) for column in range(4) for row in (0, 1)
    )
    groups = (
        WinningGroup("A", tuple(pos for pos in positions if pos // 6 == 0)),
        WinningGroup("B", tuple(pos for pos in positions if pos // 6 == 1)),
    )
    selections = select_cargo_columns(board, groups, config)
    assert [selection.column for selection in selections] == [0, 1]
    assert len(selections) == 2


def test_wild_positions_contribute_to_win_but_never_create_cargo(config):
    board = board_with_columns(
        "AWCDE",
        "AWCDE",
        "ABCDE",
        "ABCDE",
        "ABCDE",
        "ABCDE",
    )
    group = WinningGroup(
        "A",
        (
            flat_position(0, 0),
            flat_position(0, 1),
            flat_position(1, 0),
            flat_position(1, 1),
        ),
    )
    assert select_cargo_columns(board, (group,), config) == ()


def test_each_load_ladder_level_and_transient_full_state(config):
    selection = CargoSelection(2, (2, 8, 14), 3, 1)
    levels_1, transitions_1 = load_selected_columns((0, 0, 0, 0, 0, 0), (selection,))
    levels_2, transitions_2 = load_selected_columns(levels_1, (selection,))
    levels_3, transitions_3 = load_selected_columns(levels_2, (selection,))
    assert transitions_1[0].level_after == 1 and levels_1[2] == 1
    assert transitions_2[0].level_after == 2 and levels_2[2] == 2
    assert transitions_3[0].level_after == 3 and levels_3[2] == 2

    board = regular_board()
    board_1, changes_1 = apply_level_modifier(board, transitions_1[0], config)
    board_2, changes_2 = apply_level_modifier(board, transitions_2[0], config)
    assert len(changes_1) == 1 and board_1[2].count("W") == 1
    assert len(changes_2) == 2 and board_2[2].count("W") == 2


def test_adjacent_columns_couple_and_nonadjacent_columns_stay_single():
    assert group_completed_columns((2, 3)) == ((2, 3),)
    assert group_completed_columns((1, 4)) == ((1,), (4,))


def test_departure_modifier_fills_every_row_and_events_carry_authoritative_sum(config):
    groups = make_departure_groups((2, 3), "mainline", {2: 12, 3: 20}, 7)
    assert groups[0].kind == "coupled"
    assert groups[0].multiplier == 32
    board, changes = apply_departure_modifier(regular_board(), groups, config)
    assert board[2] == ("W",) * 5 and board[3] == ("W",) * 5
    assert len(changes) == 10

    ledger = EventLedger(config.wincap_units)
    event = ledger.departure_prepare(groups[0])
    assert event["components"] == [12, 20]
    assert event["multiplier"] == 32


def test_base_departure_and_round_completion_reset_levels(machine, config):
    state, ledger = machine.new_base_round()
    state = replace(state, column_levels=(2, 1, 2, 2, 0, 1))
    groups = make_departure_groups((2, 3), "yard", {2: 2, 3: 3}, 1)
    state = machine.resolve_departures(state, ledger, groups)
    assert state.column_levels == (2, 1, 0, 0, 0, 1)
    completed = machine.complete_round(state, ledger)
    assert completed.column_levels == (0, 0, 0, 0, 0, 0)
    assert ledger.events[-1]["finalLevels"] == [0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("stage", "departures", "expected_reset", "expected_stage"),
    (("yard", 0, 0, "yard"), ("mainline", 2, 1, "mainline"), ("redline", 4, 2, "redline")),
)
def test_bonus_departure_stage_resets(
    machine, config, stage, departures, expected_reset, expected_stage
):
    state = RoundState(
        mode=config.freegame_type,
        column_levels=(2, 0, 2, 0, 1, 0),
        departures=departures,
        stage=stage,
        free_spins_remaining=5,
    )
    ledger = EventLedger(config.wincap_units)
    group = make_departure_groups((2,), stage, {2: 6}, departures + 1)
    next_state = machine.resolve_departures(state, ledger, group)
    assert next_state.column_levels[2] == expected_reset
    assert next_state.stage == expected_stage


@pytest.mark.parametrize(
    ("departures", "stage", "expected_reset", "stage_after"),
    ((1, "yard", 0, "mainline"), (3, "mainline", 1, "redline")),
)
def test_bonus_stage_advances_only_after_current_departure_resolves(
    machine, config, departures, stage, expected_reset, stage_after
):
    state = RoundState(
        mode=config.freegame_type,
        column_levels=(0, 0, 2, 0, 0, 0),
        departures=departures,
        stage=stage,
        free_spins_remaining=3,
    )
    ledger = EventLedger(config.wincap_units)
    group = make_departure_groups((2,), stage, {2: 6}, departures + 1)
    next_state = machine.resolve_departures(state, ledger, group)
    assert next_state.column_levels[2] == expected_reset
    assert next_state.stage == stage_after
    assert ledger.events[-1]["stageAfter"] == stage_after


def test_bonus_state_persists_across_free_spins_and_retrigger(machine, config):
    base, ledger = machine.new_base_round()
    base = replace(base, column_levels=(1, 0, 2, 1, 0, 0))
    bonus = machine.trigger_natural_bonus(base, ledger, (0, 6, 12, 18))
    first = machine.start_free_spin(bonus, ledger)
    retriggered = machine.retrigger(first, ledger, (1, 7, 13, 19))
    second = machine.start_free_spin(retriggered, ledger)
    assert bonus.column_levels == first.column_levels == retriggered.column_levels == second.column_levels
    assert retriggered.stage == first.stage
    assert retriggered.departures == first.departures
    assert retriggered.free_spins_remaining == first.free_spins_remaining + 4
    assert event_types(ledger.events) == [
        "bonus_trigger",
        "free_spin_start",
        "retrigger",
        "free_spin_start",
    ]


def test_exact_cap_emits_terminal_max_win(config):
    ledger = EventLedger(config.wincap_units)
    win = ledger.win_result(config.wincap_units)
    assert win["stepPayoutUnits"] == config.wincap_units
    assert event_types(ledger.events) == ["win_result", "max_win"]
    assert ledger.capped is True


def test_over_cap_is_truncated_in_integer_units(config):
    ledger = EventLedger(config.wincap_units, starting_payout_units=config.wincap_units - 75)
    win = ledger.win_result(1_000)
    assert win["requestedStepPayoutUnits"] == 1_000
    assert win["stepPayoutUnits"] == 75
    assert win["roundPayoutAfterUnits"] == config.wincap_units
    assert ledger.events[-1] == {
        "index": 1,
        "type": "max_win",
        "amountUnits": config.wincap_units,
    }


def test_no_payout_event_can_follow_max_win(config):
    ledger = EventLedger(config.wincap_units)
    ledger.win_result(config.wincap_units)
    with pytest.raises(TerminalPayoutError):
        ledger.win_result(1)
    with pytest.raises(TerminalPayoutError):
        ledger.retrigger((0, 6, 12, 18), 4, 7)
    assert event_types(ledger.events) == ["win_result", "max_win"]
    ledger.round_complete((0, 0, 0, 0, 0, 0))
    assert event_types(ledger.events) == ["win_result", "max_win", "round_complete"]


def test_seed_reset_reproduces_same_trace_and_separates_different_seed(machine):
    machine.reset_seed(9, 12345)
    first = machine.deterministic_trace()
    machine.reset_seed(9, 12345)
    second = machine.deterministic_trace()
    machine.reset_seed(9, 12346)
    third = machine.deterministic_trace()
    assert first == second
    assert first != third
    machine.reset_seed(77)
    by_sim = machine.deterministic_trace()
    machine.reset_seed(77)
    assert machine.deterministic_trace() == by_sim
    machine.reset_seed(78)
    assert machine.deterministic_trace() != by_sim


def test_capped_contribution_ledger_remains_additive(config):
    ledger = EventLedger(config.wincap_units)
    buckets = (
        "base_plain",
        "base_modifier",
        "base_single_departure",
        "base_coupled_departure",
        "bonus_yard",
        "bonus_mainline",
    )
    for bucket in buckets:
        ledger.board_reveal(regular_board())
        ledger.win_result(100, contribution_bucket=bucket)
        ledger.symbols_remove((0,))
        ledger.columns_load(())
        ledger.cascade_refill(regular_board())
        ledger.no_win()
    ledger.board_reveal(regular_board())
    ledger.win_result(
        config.wincap_units,
        contribution_bucket="bonus_redline",
        applied_departure_ids=("dep-0001",),
    )
    assert sum(ledger.contribution_units.values()) == config.wincap_units
    assert ledger.contribution_units["base_plain"] == 100
    assert ledger.contribution_units["base_modifier"] == 100
    assert ledger.contribution_units["bonus_redline"] == config.wincap_units - 600
    ledger.round_complete((2, 0, 1, 0, 0, 0))
    validate_contract(ledger.events, config.wincap_units)


def test_cascade_refill_is_deep_authoritative_board(config):
    source = [list(column) for column in regular_board()]
    ledger = EventLedger(config.wincap_units)
    event = ledger.cascade_refill(source)
    source[0][0] = "S"
    assert event["resultingBoard"][0][0] == "A"


def test_custom_validator_rejects_payout_after_max_win(config):
    ledger = EventLedger(config.wincap_units)
    ledger.board_reveal(regular_board())
    ledger.win_result(config.wincap_units)
    invalid = [dict(event) for event in ledger.events]
    invalid.append(
        {
            "index": 3,
            "type": "win_result",
            "requestedStepPayoutUnits": 1,
            "stepPayoutUnits": 0,
            "roundPayoutBeforeUnits": config.wincap_units,
            "roundPayoutAfterUnits": config.wincap_units,
            "contributionBucket": "base_plain",
        }
    )
    with pytest.raises(ValueError, match="after max_win"):
        validate_contract(invalid, config.wincap_units)


def test_forced_bonus_path_is_separate_from_natural_frequency_path(machine, config):
    state, natural_ledger = machine.new_base_round()
    machine.trigger_natural_bonus(state, natural_ledger, (0, 6, 12, 18))
    assert natural_ledger.events[-1]["outcomePath"] == "natural"

    state, forced_ledger = machine.new_base_round()
    forced = machine.build_forced_bonus(
        state, forced_ledger, (0, 6, 12, 18), "forced_contract"
    )
    assert forced.mode == config.freegame_type
    assert forced_ledger.events[-1]["outcomePath"] == "forced"
    with pytest.raises(ValueError, match="dedicated forced criterion"):
        machine.build_forced_bonus(
            state, EventLedger(config.wincap_units), (0, 6, 12, 18), "natural_bonus"
        )
    machine.new_base_round()
    assert machine.active_outcome_path == "natural"


def test_bonus_completion_cannot_leak_state_into_next_bet(machine, config):
    state, ledger = machine.new_base_round()
    state = replace(state, column_levels=(1, 0, 2, 1, 0, 0))
    bonus = machine.trigger_natural_bonus(state, ledger, (0, 6, 12, 18))
    completed = machine.complete_bonus(bonus, ledger, feature_payout_units=0)
    assert completed.mode == config.basegame_type
    assert completed.column_levels == (0, 0, 0, 0, 0, 0)
    assert completed.departures == 0
    assert completed.free_spins_remaining == 0


def test_custom_validator_rejects_noncanonical_event_order(config):
    invalid = [
        {"index": 0, "type": "board_reveal", "board": []},
        {"index": 1, "type": "columns_load", "transitions": []},
    ]
    with pytest.raises(ValueError, match="must follow symbols_remove"):
        validate_contract(invalid, config.wincap_units)


def test_canonical_tumble_and_departure_event_order(config):
    ledger = EventLedger(config.wincap_units)
    board = regular_board()
    transition = LoadTransition(2, 2, 3, (2, 8, 14), 3, 1)
    group = make_departure_groups((2,), "yard", {2: 3}, 1)[0]
    modified, changes = apply_departure_modifier(board, (group,), config)

    ledger.board_reveal(board)
    ledger.win_result(100)
    ledger.symbols_remove((2, 8, 14))
    ledger.columns_load((transition,))
    ledger.departure_prepare(group)
    ledger.cascade_refill(board)
    ledger.board_modifier("departure", changes, modified)
    ledger.win_result(300, applied_departure_ids=(group.departure_id,))
    ledger.departure_resolve(group, (0,), 0, "yard")
    ledger.round_complete((0, 0, 0, 0, 0, 0))

    assert event_types(ledger.events) == [
        "board_reveal",
        "win_result",
        "symbols_remove",
        "columns_load",
        "departure_prepare",
        "cascade_refill",
        "board_modifier",
        "win_result",
        "departure_resolve",
        "round_complete",
    ]
    validate_contract(ledger.events, config.wincap_units)
