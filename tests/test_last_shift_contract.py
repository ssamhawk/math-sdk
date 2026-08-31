"""Deterministic T4-C1 contract tests for Last Shift."""

from copy import deepcopy
from dataclasses import replace

import pytest

from games.last_shift.game_calculations import (
    CargoSelection,
    LoadTransition,
    WinningGroup,
    apply_departure_modifier,
    apply_level_modifier,
    derive_contribution_bucket,
    derive_scatter_positions,
    evaluate_pay_anywhere,
    flat_position,
    group_completed_columns,
    load_selected_columns,
    make_departure_groups,
    select_cargo_columns,
    serialize_wins,
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


def cargo_board():
    return board_with_columns(
        "BCDEF",
        "GHBCD",
        "AAAAA",
        "AAAAA",
        "EFGHB",
        "CDEFG",
    )


def scatter_board():
    board = [list(column) for column in regular_board()]
    for position in (0, 6, 12, 18):
        board[position % 6][position // 6] = "S"
    return tuple(tuple(column) for column in board)


def five_scatter_board():
    board = [list(column) for column in scatter_board()]
    board[0][4] = "S"
    return tuple(tuple(column) for column in board)


def all_wild_board():
    return board_with_columns(*(["WWWWW"] * 6))


def balanced_loss_board():
    symbols = tuple("ABCDEFGH")
    return tuple(
        tuple(symbols[(row * 6 + column) % len(symbols)] for row in range(5))
        for column in range(6)
    )


def cargo_scatter_board():
    board = [list(column) for column in cargo_board()]
    for position in (0, 1, 4, 5):
        board[position % 6][position // 6] = "S"
    return tuple(tuple(column) for column in board)


def wild_only_win_board():
    symbols = ["W"] * 8 + ["S"] * 22
    return tuple(
        tuple(symbols[row * 6 + column] for row in range(5))
        for column in range(6)
    )


def cargo_positions():
    return tuple(
        flat_position(column, row) for column in (2, 3) for row in range(5)
    )


def payout_groups():
    return [{"symbol": "A", "positions": list(cargo_positions())}]


def apply_scheduled_modifiers(board, transitions, config):
    resulting = board
    changes = []
    for transition in transitions:
        resulting, transition_changes = apply_level_modifier(
            resulting, transition, config
        )
        changes.extend(transition_changes)
    return resulting, tuple(changes)


def enter_natural_bonus(machine, state, ledger, board):
    positions = derive_scatter_positions(board, machine.config)
    ledger.board_reveal(board)
    ledger.no_win()
    return machine.trigger_natural_bonus(state, ledger, board, positions)


def finish_bonus_with_losses(machine, state, ledger):
    while state.free_spins_remaining:
        state = machine.start_free_spin(state, ledger)
        ledger.board_reveal(regular_board())
        ledger.no_win()
        ledger.free_spin_complete(0, state.free_spins_remaining, state.column_levels)
    return machine.complete_bonus(state, ledger, ledger.feature_payout_units)


def build_latched_cascade_bonus(machine, later_board=regular_board()):
    config = machine.config
    state, ledger = machine.new_base_round()
    first_board = wild_only_win_board()
    ledger.board_reveal(first_board)
    evaluated, payout = evaluate_pay_anywhere(first_board, config)
    ledger.win_result(payout, groups=serialize_wins(evaluated))
    ledger.symbols_remove(sorted({position for win in evaluated for position in win.positions}))
    ledger.columns_load(())
    ledger.cascade_refill(later_board)
    ledger.no_win()
    state = machine.trigger_natural_bonus(
        state,
        ledger,
        first_board,
        derive_scatter_positions(first_board, config),
    )
    finish_bonus_with_losses(machine, state, ledger)
    return ledger.events


def emit_three_level_departure(
    machine,
    config,
    state,
    ledger,
    initial_board=None,
    departure_board=None,
    components=None,
):
    board = initial_board or cargo_board()
    departure_board = departure_board or cargo_board()
    components = components or ({2: 2, 3: 3} if state.mode == "basegame" else {2: 6, 3: 10})
    ledger.board_reveal(board)
    completed_group = None
    modifier_reason = None

    for level in (1, 2, 3):
        group = WinningGroup("A", cargo_positions())
        evaluated, payout = evaluate_pay_anywhere(board, config)
        ledger.win_result(
            payout,
            groups=serialize_wins(evaluated),
            contribution_bucket=derive_contribution_bucket(
                state.mode, modifier_reason, ()
            ),
        )
        winning_positions = sorted({p for win in evaluated for p in win.positions})
        ledger.symbols_remove(winning_positions)
        selections = select_cargo_columns(
            board,
            tuple(WinningGroup(win.symbol, win.positions) for win in evaluated),
            config,
        )
        next_levels, transitions = load_selected_columns(
            state.column_levels, selections
        )
        ledger.columns_load(transitions)
        state = replace(state, column_levels=next_levels)

        if level < 3:
            ledger.cascade_refill(cargo_board())
            board, changes = apply_scheduled_modifiers(
                cargo_board(), transitions, config
            )
            ledger.board_modifier(f"load_level_{level}", changes, board)
            modifier_reason = f"load_level_{level}"
            continue

        completed_group = make_departure_groups(
            (2, 3), state.stage, components, 1, mode=state.mode
        )[0]
        ledger.departure_prepare(completed_group)
        ledger.cascade_refill(departure_board)
        board, changes = apply_departure_modifier(
            departure_board, (completed_group,), config
        )
        ledger.board_modifier("departure", changes, board)
        evaluated, payout = evaluate_pay_anywhere(board, config, (completed_group,))
        ledger.win_result(
            payout,
            groups=serialize_wins(evaluated),
            applied_departure_ids=(completed_group.departure_id,),
            contribution_bucket=derive_contribution_bucket(
                state.mode, "departure", (completed_group,)
            ),
        )
        if not ledger.capped:
            state = machine.resolve_departures(state, ledger, (completed_group,))

    return state, completed_group


def build_end_to_end_departure_fixture(machine, config):
    state, ledger = machine.new_base_round()
    state, completed_group = emit_three_level_departure(
        machine, config, state, ledger
    )

    final_state = machine.complete_round(state, ledger)
    return ledger.events, final_state, completed_group


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
    groups = make_departure_groups((2, 3), "mainline", {2: 12, 3: 20}, 7, mode="freegame")
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
    component = {"yard": 6, "mainline": 12, "redline": 25}[stage]
    group = make_departure_groups((2,), stage, {2: component}, departures + 1, mode="freegame")
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
    component = {"yard": 6, "mainline": 12}[stage]
    group = make_departure_groups((2,), stage, {2: component}, departures + 1, mode="freegame")
    next_state = machine.resolve_departures(state, ledger, group)
    assert next_state.column_levels[2] == expected_reset
    assert next_state.stage == stage_after
    assert ledger.events[-1]["stageAfter"] == stage_after


def test_bonus_state_persists_across_free_spins_and_retrigger(machine, config):
    base, ledger = machine.new_base_round()
    base = replace(base, column_levels=(1, 0, 2, 1, 0, 0))
    bonus = enter_natural_bonus(machine, base, ledger, scatter_board())
    first = machine.start_free_spin(bonus, ledger)
    ledger.board_reveal(scatter_board())
    ledger.no_win()
    retriggered = machine.retrigger(first, ledger, scatter_board(), (0, 6, 12, 18))
    ledger.free_spin_complete(
        0, retriggered.free_spins_remaining, retriggered.column_levels
    )
    second = machine.start_free_spin(retriggered, ledger)
    assert bonus.column_levels == first.column_levels == retriggered.column_levels == second.column_levels
    assert retriggered.stage == first.stage
    assert retriggered.departures == first.departures
    assert retriggered.free_spins_remaining == first.free_spins_remaining + 4
    assert event_types(ledger.events) == [
        "round_start",
        "board_reveal",
        "no_win",
        "bonus_trigger",
        "free_spin_start",
        "board_reveal",
        "no_win",
        "retrigger",
        "free_spin_complete",
        "free_spin_start",
    ]


def test_exact_cap_emits_terminal_max_win(config):
    ledger = EventLedger(config.wincap_units)
    win = ledger.win_result(config.wincap_units)
    assert win["stepPayoutUnits"] == config.wincap_units
    assert event_types(ledger.events) == ["win_result", "max_win"]
    assert ledger.capped is True


def test_over_cap_is_truncated_in_integer_units(config):
    ledger = EventLedger(config.wincap_units, starting_payout_units=config.wincap_units - 100)
    win = ledger.win_result(1_000)
    assert win["requestedStepPayoutUnits"] == 1_000
    assert win["stepPayoutUnits"] == 100
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
        ledger.retrigger((0, 6, 12, 18), 4, 7, (0, 0, 0, 0, 0, 0), "yard", 0, scatter_board())
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
        "bonus_plain",
        "bonus_modifier",
    )
    for bucket in buckets:
        ledger.win_result(100, groups=payout_groups(), contribution_bucket=bucket)
    ledger.win_result(
        config.wincap_units,
        groups=payout_groups(),
        contribution_bucket="bonus_redline_departure",
    )
    assert sum(ledger.contribution_units.values()) == config.wincap_units
    assert ledger.contribution_units["base_plain"] == 100
    assert ledger.contribution_units["base_modifier"] == 100
    assert ledger.contribution_units["bonus_redline_departure"] == config.wincap_units - 600


def test_cascade_refill_is_deep_authoritative_board(config):
    source = [list(column) for column in regular_board()]
    ledger = EventLedger(config.wincap_units)
    event = ledger.cascade_refill(source)
    source[0][0] = "S"
    assert event["resultingBoard"][0][0] == "A"


def test_custom_validator_rejects_payout_after_max_win(config):
    ledger = EventLedger(config.wincap_units)
    ledger.round_start("basegame", (0, 0, 0, 0, 0, 0), 0, "yard", 0)
    ledger.board_reveal(cargo_board())
    ledger.win_result(
        config.wincap_units, groups=payout_groups(), contribution_bucket="base_plain"
    )
    ledger.round_complete((0, 0, 0, 0, 0, 0))
    invalid = deepcopy(ledger.events)
    invalid.insert(-1, {"type": "board_reveal", "board": [list(c) for c in cargo_board()]})
    for index, event in enumerate(invalid):
        event["index"] = index
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_forced_bonus_path_is_separate_from_natural_frequency_path(machine, config):
    state, natural_ledger = machine.new_base_round()
    enter_natural_bonus(machine, state, natural_ledger, scatter_board())
    assert natural_ledger.events[-1]["outcomePath"] == "natural"

    state, forced_ledger = machine.new_base_round()
    forced = machine.build_forced_bonus(
        state, forced_ledger, scatter_board(), (0, 6, 12, 18), "forced_contract"
    )
    assert forced.mode == config.freegame_type
    assert forced_ledger.events[-1]["outcomePath"] == "forced"
    with pytest.raises(ValueError, match="dedicated forced criterion"):
        machine.build_forced_bonus(
            state, EventLedger(config.wincap_units), scatter_board(), (0, 6, 12, 18), "natural_bonus"
        )
    machine.new_base_round()
    assert machine.active_outcome_path == "natural"


def test_bonus_completion_cannot_leak_state_into_next_bet(machine, config):
    state, ledger = machine.new_base_round()
    state = replace(state, column_levels=(1, 0, 2, 1, 0, 0))
    bonus = enter_natural_bonus(machine, state, ledger, scatter_board())
    completed = finish_bonus_with_losses(machine, bonus, ledger)
    assert completed.mode == config.basegame_type
    assert completed.column_levels == (0, 0, 0, 0, 0, 0)
    assert completed.departures == 0
    assert completed.free_spins_remaining == 0


def test_end_to_end_board_to_departure_trace(machine, config):
    events, final_state, group = build_end_to_end_departure_fixture(machine, config)
    validate_contract(events, config.wincap_units)
    assert group.kind == "coupled" and group.columns == (2, 3)
    assert final_state.column_levels == (0, 0, 0, 0, 0, 0)
    assert final_state.round_payout_units == events[-1]["finalPayoutUnits"]
    assert final_state.round_payout_units != 600


def test_pay_anywhere_evaluator_requires_eight_and_derives_integer_payout(config):
    below = board_with_columns("WSSSS", "WSSSS", "WSSSS", "WSSSS", "WSSSS", "WSSSS")
    wins, payout = evaluate_pay_anywhere(below, config)
    assert wins == () and payout == 0
    wins, payout = evaluate_pay_anywhere(wild_only_win_board(), config)
    assert wins and all(win.count >= 8 for win in wins)
    assert payout == sum(win.payout_units for win in wins)
    assert all(isinstance(win.payout_units, int) for win in wins)


def test_refill_can_go_directly_to_no_win_without_modifier(config):
    ledger = EventLedger(config.wincap_units)
    ledger.round_start("basegame", (0, 0, 0, 0, 0, 0), 0, "yard", 0)
    board = all_wild_board()
    evaluated, payout = evaluate_pay_anywhere(board, config)
    positions = sorted({position for win in evaluated for position in win.positions})
    ledger.board_reveal(board)
    ledger.win_result(payout, groups=serialize_wins(evaluated))
    ledger.symbols_remove(positions)
    ledger.columns_load(())
    ledger.cascade_refill(regular_board())
    ledger.no_win()
    ledger.round_complete((0, 0, 0, 0, 0, 0))
    validate_contract(ledger.events, config.wincap_units)


def test_duplicate_selected_columns_are_rejected():
    selections = (
        CargoSelection(2, (2, 8), 2, 1),
        CargoSelection(2, (14, 20), 2, 2),
    )
    with pytest.raises(ValueError, match="selected columns must be unique"):
        load_selected_columns((0, 0, 0, 0, 0, 0), selections)


@pytest.mark.parametrize(
    ("departures", "stage", "expected_departures", "expected_stage", "reset"),
    ((0, "yard", 2, "mainline", 0), (2, "mainline", 4, "redline", 1)),
)
def test_coupled_departure_counts_completed_columns_at_thresholds(
    machine, config, departures, stage, expected_departures, expected_stage, reset
):
    state = RoundState(
        mode=config.freegame_type,
        column_levels=(0, 0, 2, 2, 0, 0),
        departures=departures,
        stage=stage,
        free_spins_remaining=3,
    )
    ledger = EventLedger(config.wincap_units)
    components = {"yard": {2: 6, 3: 10}, "mainline": {2: 12, 3: 20}}[stage]
    group = make_departure_groups((2, 3), stage, components, 1, mode="freegame")
    next_state = machine.resolve_departures(state, ledger, group)
    assert next_state.departures == expected_departures
    assert next_state.stage == expected_stage
    assert next_state.column_levels[2:4] == (reset, reset)


def test_resolve_rejects_stage_at_selection_mismatch(machine, config):
    state = RoundState(
        mode=config.freegame_type,
        column_levels=(0, 0, 2, 0, 0, 0),
        departures=2,
        stage="mainline",
        free_spins_remaining=2,
    )
    group = make_departure_groups((2,), "yard", {2: 6}, 1, mode="freegame")
    with pytest.raises(ValueError, match="stageAtSelection"):
        machine.resolve_departures(state, EventLedger(config.wincap_units), group)


@pytest.mark.parametrize("positions", ((0, 0, 6, 12), (0, 6, 12, 30)))
def test_bonus_and_retrigger_positions_must_be_unique_and_in_range(
    machine, config, positions
):
    state, ledger = machine.new_base_round()
    with pytest.raises(ValueError, match="positions|position"):
        machine.trigger_natural_bonus(state, ledger, scatter_board(), positions)
    with pytest.raises(ValueError, match="positions|position"):
        ledger.bonus_trigger(positions, 10, (0, 0, 0, 0, 0, 0), board=scatter_board())


def test_bonus_positions_must_point_to_scatter(machine, config):
    state, ledger = machine.new_base_round()
    with pytest.raises(ValueError, match="scatter set"):
        machine.trigger_natural_bonus(state, ledger, regular_board(), (0, 6, 12, 18))
    with pytest.raises(ValueError, match="point to S"):
        ledger.bonus_trigger(
            (0, 6, 12, 18), 10, (0, 0, 0, 0, 0, 0), board=regular_board()
        )


@pytest.mark.parametrize(
    ("departures", "stage", "components", "expected_stage", "reset"),
    (
        (0, "yard", {1: 6, 4: 10}, "mainline", 0),
        (2, "mainline", {1: 12, 4: 20}, "redline", 1),
    ),
)
def test_nonadjacent_simultaneous_departures_freeze_selection_stage(
    machine, config, departures, stage, components, expected_stage, reset
):
    state = RoundState(
        mode=config.freegame_type,
        column_levels=(0, 2, 0, 0, 2, 0),
        departures=departures,
        stage=stage,
        free_spins_remaining=2,
    )
    groups = make_departure_groups(
        (1, 4), stage, components, 1, mode="freegame"
    )
    assert len(groups) == 2 and all(group.stage_at_selection == stage for group in groups)
    resolved = machine.resolve_departures(
        state, EventLedger(config.wincap_units), groups
    )
    assert resolved.departures == departures + 2
    assert resolved.stage == expected_stage
    assert resolved.column_levels[1] == resolved.column_levels[4] == reset


def test_provisional_paytable_obeys_sdk_payout_quantum(config):
    assert config.paytable_units
    assert all(
        payout >= config.payout_quantum_units
        and payout % config.payout_quantum_units == 0
        for payout in config.paytable_units.values()
    )
    with pytest.raises(ValueError, match="payout quantum"):
        EventLedger(config.wincap_units).win_result(15, groups=payout_groups())


def test_complete_bonus_state_and_payout_reconcile(machine, config):
    state, ledger = machine.new_base_round()
    ledger.board_reveal(scatter_board())
    ledger.no_win()
    state = machine.trigger_natural_bonus(state, ledger, scatter_board(), (0, 6, 12, 18))
    for _ in range(config.initial_free_spins):
        state = machine.start_free_spin(state, ledger)
        ledger.board_reveal(regular_board())
        ledger.no_win()
        ledger.free_spin_complete(0, state.free_spins_remaining, state.column_levels)
    machine.complete_bonus(state, ledger, 0)
    validate_contract(ledger.events, config.wincap_units)


def reindex(events):
    for index, event in enumerate(events):
        event["index"] = index
    return events


def build_base_loss_events(config, board=regular_board()):
    ledger = EventLedger(config.wincap_units)
    ledger.round_start("basegame", (0, 0, 0, 0, 0, 0), 0, "yard", 0)
    ledger.board_reveal(board)
    ledger.no_win()
    ledger.round_complete((0, 0, 0, 0, 0, 0))
    return ledger.events


def test_no_win_is_derived_from_authoritative_post_modifier_board(machine, config):
    ledger = EventLedger(config.wincap_units)
    ledger.round_start("basegame", (0, 0, 0, 0, 0, 0), 0, "yard", 0)
    ledger.board_reveal(all_wild_board())
    with pytest.raises(ValueError, match="zero-payout"):
        ledger.no_win()

    invalid = deepcopy(build_base_loss_events(config))
    invalid[1]["board"] = [list(column) for column in all_wild_board()]
    with pytest.raises(ValueError, match="zero-payout"):
        validate_contract(invalid, config.wincap_units)

    producer = EventLedger(config.wincap_units)
    producer.round_start("basegame", (0, 0, 0, 0, 0, 0), 0, "yard", 0)
    producer.board_reveal(regular_board())
    producer.board_modifier("load_level_1", (), all_wild_board())
    with pytest.raises(ValueError, match="zero-payout"):
        producer.no_win()

    winning_modified, _, _ = build_end_to_end_departure_fixture(machine, config)
    modifier_index = next(
        index
        for index, event in enumerate(winning_modified)
        if event["type"] == "board_modifier"
    )
    invalid = deepcopy(winning_modified)
    before = invalid[modifier_index + 1]["roundPayoutBeforeUnits"]
    invalid[modifier_index + 1] = {
        "index": modifier_index + 1,
        "type": "no_win",
        "roundPayoutUnits": before,
    }
    with pytest.raises(ValueError, match="zero-payout"):
        validate_contract(invalid, config.wincap_units)


def test_base_scatter_latch_requires_trigger_at_terminal_boundary(machine, config):
    state, ledger = machine.new_base_round()
    ledger.board_reveal(scatter_board())
    ledger.no_win()
    with pytest.raises(ValueError, match="latched scatter"):
        machine.complete_round(state, ledger)

    invalid = deepcopy(build_base_loss_events(config))
    invalid[1]["board"] = [list(column) for column in scatter_board()]
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_free_scatter_latch_requires_retrigger(machine, config):
    state, ledger = machine.new_base_round()
    state = enter_natural_bonus(machine, state, ledger, scatter_board())
    state = machine.start_free_spin(state, ledger)
    ledger.board_reveal(scatter_board())
    ledger.no_win()
    with pytest.raises(ValueError, match="latched scatter"):
        ledger.free_spin_complete(0, state.free_spins_remaining, state.column_levels)

    valid_state, valid_ledger = machine.new_base_round()
    valid_state = enter_natural_bonus(machine, valid_state, valid_ledger, scatter_board())
    finish_bonus_with_losses(machine, valid_state, valid_ledger)
    invalid = deepcopy(valid_ledger.events)
    first_free_start = next(
        index for index, event in enumerate(invalid) if event["type"] == "free_spin_start"
    )
    invalid[first_free_start + 1]["board"] = [list(column) for column in scatter_board()]
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_trigger_reports_complete_five_scatter_latch(machine, config):
    state, ledger = machine.new_base_round()
    state = enter_natural_bonus(machine, state, ledger, five_scatter_board())
    finish_bonus_with_losses(machine, state, ledger)
    validate_contract(ledger.events, config.wincap_units)
    trigger = next(event for event in ledger.events if event["type"] == "bonus_trigger")
    assert trigger["positions"] == list(derive_scatter_positions(five_scatter_board(), config))

    invalid = deepcopy(ledger.events)
    next(event for event in invalid if event["type"] == "bonus_trigger")["positions"] = [0, 6, 12, 18]
    with pytest.raises(ValueError, match="complete scatter latch"):
        validate_contract(invalid, config.wincap_units)


def test_first_scatter_latch_survives_cascade_and_cannot_be_replaced(machine, config):
    events = build_latched_cascade_bonus(machine, later_board=scatter_board())
    validate_contract(events, config.wincap_units)
    first_positions = list(derive_scatter_positions(wild_only_win_board(), config))
    assert next(event for event in events if event["type"] == "bonus_trigger")["positions"] == first_positions

    omitted = deepcopy(events)
    terminal_loss = next(
        index
        for index, event in enumerate(omitted)
        if event["type"] == "no_win" and index > 1
    )
    final = deepcopy(omitted[-1])
    omitted = reindex(omitted[: terminal_loss + 1] + [final])
    with pytest.raises(ValueError):
        validate_contract(omitted, config.wincap_units)

    replaced = deepcopy(events)
    next(event for event in replaced if event["type"] == "bonus_trigger")["positions"] = [0, 6, 12, 18]
    with pytest.raises(ValueError, match="first complete scatter latch"):
        validate_contract(replaced, config.wincap_units)


def test_second_trigger_in_same_spin_is_rejected(machine, config):
    events = build_latched_cascade_bonus(machine)
    invalid = deepcopy(events)
    trigger_index = next(
        index for index, event in enumerate(invalid) if event["type"] == "bonus_trigger"
    )
    invalid.insert(trigger_index + 1, deepcopy(invalid[trigger_index]))
    reindex(invalid)
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_scatter_departure_requires_base_trigger_after_final_resolve(machine, config):
    state, ledger = machine.new_base_round()
    state, _ = emit_three_level_departure(
        machine, config, state, ledger, initial_board=cargo_scatter_board()
    )
    resolve_index = len(ledger.events) - 1
    state = machine.trigger_natural_bonus(
        state,
        ledger,
        cargo_scatter_board(),
        derive_scatter_positions(cargo_scatter_board(), config),
    )
    finish_bonus_with_losses(machine, state, ledger)
    validate_contract(ledger.events, config.wincap_units)

    invalid = reindex(deepcopy(ledger.events[: resolve_index + 1]) + [deepcopy(ledger.events[-1])])
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_scatter_departure_requires_free_retrigger_after_final_resolve(machine, config):
    state, ledger = machine.new_base_round()
    state = enter_natural_bonus(machine, state, ledger, scatter_board())
    state = machine.start_free_spin(state, ledger)
    state, _ = emit_three_level_departure(
        machine, config, state, ledger, initial_board=cargo_scatter_board()
    )
    state = machine.retrigger(
        state,
        ledger,
        cargo_scatter_board(),
        derive_scatter_positions(cargo_scatter_board(), config),
    )
    first_spin_start_payout = next(
        event["roundPayoutUnits"]
        for event in ledger.events
        if event["type"] == "round_start"
    )
    ledger.free_spin_complete(
        ledger.round_payout_units - first_spin_start_payout,
        state.free_spins_remaining,
        state.column_levels,
    )
    finish_bonus_with_losses(machine, state, ledger)
    validate_contract(ledger.events, config.wincap_units)

    invalid = deepcopy(ledger.events)
    invalid.remove(next(event for event in invalid if event["type"] == "retrigger"))
    reindex(invalid)
    with pytest.raises(ValueError):
        validate_contract(invalid, config.wincap_units)


def test_modifier_attribution_is_consumed_by_no_win(machine, config):
    state, ledger = machine.new_base_round()
    state = enter_natural_bonus(machine, state, ledger, scatter_board())
    state = machine.start_free_spin(state, ledger)
    spin_start_payout = ledger.round_payout_units
    board = cargo_board()
    ledger.board_reveal(board)
    evaluated, payout = evaluate_pay_anywhere(board, config)
    ledger.win_result(payout, groups=serialize_wins(evaluated), contribution_bucket="bonus_plain")
    ledger.symbols_remove(sorted({position for win in evaluated for position in win.positions}))
    selections = select_cargo_columns(
        board, tuple(WinningGroup(win.symbol, win.positions) for win in evaluated), config
    )
    next_levels, transitions = load_selected_columns(state.column_levels, selections)
    state = replace(state, column_levels=next_levels)
    ledger.columns_load(transitions)
    ledger.cascade_refill(balanced_loss_board())
    modified, changes = apply_scheduled_modifiers(
        balanced_loss_board(), transitions, config
    )
    assert evaluate_pay_anywhere(modified, config)[1] == 0
    ledger.board_modifier("load_level_1", changes, modified)
    ledger.no_win()
    ledger.free_spin_complete(
        ledger.round_payout_units - spin_start_payout,
        state.free_spins_remaining,
        state.column_levels,
    )

    state = machine.start_free_spin(state, ledger)
    spin_start_payout = ledger.round_payout_units
    board = wild_only_win_board()
    ledger.board_reveal(board)
    evaluated, payout = evaluate_pay_anywhere(board, config)
    ledger.win_result(payout, groups=serialize_wins(evaluated), contribution_bucket="bonus_plain")
    second_win_index = len(ledger.events) - 1
    ledger.symbols_remove(sorted({position for win in evaluated for position in win.positions}))
    ledger.columns_load(())
    ledger.cascade_refill(regular_board())
    ledger.no_win()
    state = machine.retrigger(
        state, ledger, board, derive_scatter_positions(board, config)
    )
    ledger.free_spin_complete(
        ledger.round_payout_units - spin_start_payout,
        state.free_spins_remaining,
        state.column_levels,
    )
    finish_bonus_with_losses(machine, state, ledger)
    validate_contract(ledger.events, config.wincap_units)

    invalid = deepcopy(ledger.events)
    invalid[second_win_index]["contributionBucket"] = "bonus_modifier"
    with pytest.raises(ValueError, match="contribution bucket"):
        validate_contract(invalid, config.wincap_units)


def test_forced_outcome_path_is_rejected_by_release_validator(machine, config):
    state, ledger = machine.new_base_round()
    ledger.board_reveal(scatter_board())
    ledger.no_win()
    state = machine.build_forced_bonus(
        state, ledger, scatter_board(), (0, 6, 12, 18), "forced_contract"
    )
    finish_bonus_with_losses(machine, state, ledger)
    with pytest.raises(ValueError, match="forced outcomePath"):
        validate_contract(ledger.events, config.wincap_units)


def test_noncap_bonus_completion_guards_and_ledger_payout(machine, config):
    state, ledger = machine.new_base_round()
    state = enter_natural_bonus(machine, state, ledger, scatter_board())
    with pytest.raises(ValueError, match="zero remaining spins"):
        machine.complete_bonus(state, ledger, 0)
    while state.free_spins_remaining:
        state = machine.start_free_spin(state, ledger)
        ledger.board_reveal(regular_board())
        ledger.no_win()
        ledger.free_spin_complete(0, state.free_spins_remaining, state.column_levels)
    with pytest.raises(ValueError, match="ledger-derived"):
        ledger.bonus_complete(10)


def test_capped_completion_requires_exact_tail_and_matching_state(machine, config):
    constructor_capped = EventLedger(
        config.wincap_units, starting_payout_units=config.wincap_units
    )
    with pytest.raises(TerminalPayoutError, match="win_result -> max_win"):
        constructor_capped.round_complete((0, 0, 0, 0, 0, 0))

    state, ledger = machine.new_base_round()
    ledger.win_result(config.wincap_units)
    with pytest.raises(ValueError, match="cap status"):
        machine.complete_round(state, ledger)


def test_unresolved_departure_cannot_complete_without_cap(machine, config):
    state, ledger = machine.new_base_round()
    group = make_departure_groups((2,), "yard", {2: 2}, 1)[0]
    ledger.departure_prepare(group)
    with pytest.raises(ValueError, match="unresolved"):
        machine.complete_round(state, ledger)


def test_capped_departure_cancels_all_pending_and_uses_exact_terminal_tail(machine, config):
    state, ledger = machine.new_base_round()
    state, group = emit_three_level_departure(
        machine,
        config,
        state,
        ledger,
        departure_board=all_wild_board(),
        components={2: 4, 3: 4},
    )
    assert ledger.capped
    capped_win = ledger.events[-2]
    assert capped_win["appliedDepartureIds"] == [group.departure_id]
    assert ledger.unresolved_departure_ids == set()
    state = replace(
        state,
        round_payout_units=ledger.round_payout_units,
        capped=True,
    )
    machine.complete_round(state, ledger)
    assert event_types(ledger.events[-3:]) == ["win_result", "max_win", "round_complete"]
    assert "departure_resolve" not in event_types(ledger.events)
    validate_contract(ledger.events, config.wincap_units)

    for producer in (
        lambda: ledger.free_spin_complete(0, 0, (0, 0, 0, 0, 0, 0)),
        lambda: ledger.departure_resolve(group, (0, 0), 0, "yard"),
    ):
        with pytest.raises(TerminalPayoutError):
            producer()


def mutate_departure_fixture(events, mutation):
    mutated = deepcopy(events)
    load_events = [event for event in mutated if event["type"] == "columns_load"]
    prepare = next(event for event in mutated if event["type"] == "departure_prepare")
    departure_win = next(
        event for event in mutated if event["type"] == "win_result" and event["appliedDepartureIds"]
    )
    resolve = next(event for event in mutated if event["type"] == "departure_resolve")
    if mutation == "unknown":
        mutated[1]["type"] = "mystery"
    elif mutation == "missing_final":
        mutated.pop()
    elif mutation == "start_levels":
        mutated[0]["levels"][0] = 1
    elif mutation == "duplicate_column":
        load_events[0]["transitions"][1]["column"] = 2
    elif mutation == "bad_rank":
        load_events[0]["transitions"][0]["selectionRank"] = 2
    elif mutation == "duplicate_source":
        load_events[0]["transitions"][0]["sourcePositions"][1] = load_events[0]["transitions"][0]["sourcePositions"][0]
    elif mutation == "missing_prepare":
        mutated.remove(prepare)
    elif mutation == "multiplier":
        prepare["multiplier"] += 1
    elif mutation == "nonadjacent":
        prepare["columns"] = [2, 4]
    elif mutation == "payout_id":
        departure_win["appliedDepartureIds"] = ["dep-wrong"]
    elif mutation == "resolve_id":
        resolve["departureId"] = "dep-wrong"
    elif mutation == "terminal_gameplay":
        position = mutated.index(resolve) + 1
        mutated.insert(position, {"type": "board_reveal", "board": [list(c) for c in regular_board()]})
    elif mutation == "evaluator_payout":
        first_win = next(event for event in mutated if event["type"] == "win_result")
        first_win["requestedStepPayoutUnits"] += 10
    elif mutation == "omit_top_cargo":
        load_events[0]["transitions"].pop()
    elif mutation == "modifier_change":
        modifier = next(event for event in mutated if event["type"] == "board_modifier")
        modifier["changes"][0]["from_symbol"] = "H"
    elif mutation == "illegal_component":
        prepare["components"][0] = 10
        prepare["multiplier"] = sum(prepare["components"])
    elif mutation == "wrong_bucket":
        first_win = next(event for event in mutated if event["type"] == "win_result")
        first_win["contributionBucket"] = "base_modifier"
    elif mutation == "split_adjacent":
        position = mutated.index(prepare)
        first = deepcopy(prepare)
        second = deepcopy(prepare)
        first.update(departureId="dep-split-1", kind="single", columns=[2], components=[2], multiplier=2)
        second.update(departureId="dep-split-2", kind="single", columns=[3], components=[3], multiplier=3)
        mutated[position : position + 1] = [first, second]
        departure_win["appliedDepartureIds"] = ["dep-split-1", "dep-split-2"]
        departure_win["contributionBucket"] = "base_single_departure"
        resolve_position = mutated.index(resolve)
        first_resolve = deepcopy(resolve)
        second_resolve = deepcopy(resolve)
        first_resolve.update(departureId="dep-split-1", columns=[2], resetLevels=[0])
        second_resolve.update(departureId="dep-split-2", columns=[3], resetLevels=[0])
        mutated[resolve_position : resolve_position + 1] = [first_resolve, second_resolve]
    for index, event in enumerate(mutated):
        event["index"] = index
    return mutated


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown",
        "missing_final",
        "start_levels",
        "duplicate_column",
        "bad_rank",
        "duplicate_source",
        "missing_prepare",
        "multiplier",
        "nonadjacent",
        "payout_id",
        "resolve_id",
        "terminal_gameplay",
    ),
)
def test_negative_contract_mutation_matrix(machine, config, mutation):
    events, _, _ = build_end_to_end_departure_fixture(machine, config)
    with pytest.raises(ValueError):
        validate_contract(mutate_departure_fixture(events, mutation), config.wincap_units)


@pytest.mark.parametrize(
    "mutation",
    (
        "evaluator_payout",
        "omit_top_cargo",
        "modifier_change",
        "illegal_component",
        "wrong_bucket",
        "split_adjacent",
    ),
)
def test_six_residual_negative_probes(machine, config, mutation):
    events, _, _ = build_end_to_end_departure_fixture(machine, config)
    with pytest.raises(ValueError):
        validate_contract(mutate_departure_fixture(events, mutation), config.wincap_units)
