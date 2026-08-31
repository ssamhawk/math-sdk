"""T4-C1G integration tests for the real Stake SDK GameState pipeline."""

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import games.last_shift as last_shift
from games.last_shift import GameConfig, GameState
from games.last_shift.game_events import validate_contract
from games.last_shift.run import build_release
from src.state.state import GeneralGameState


FROZEN_VALIDATE_CONTRACT_HASH = (
    "d63ec575a518aea31c23d1a5696bcb8f24000437e0f76f997342bd538568401a"
)


@pytest.fixture(autouse=True)
def restore_game_config_singleton():
    config = GameConfig()
    snapshot = deepcopy(vars(config))
    try:
        yield
    finally:
        vars(config).clear()
        vars(config).update(snapshot)


def make_game(tmp_path):
    config = GameConfig()
    config.game_id = str(tmp_path / "last_shift_contract_proof_direct")
    config.construct_paths()
    game = GameState(config)
    game.betmode = "contract_proof"
    return config, game


def run_criterion(game, criterion, sim=0, seed=12345):
    game.criteria = criterion
    game.run_spin(sim, seed)
    return deepcopy(game.book.to_json())


def assert_reconciled(config, game, book):
    events = book["events"]
    final_units = events[-1]["finalPayoutUnits"]
    scale = config.payout_scale

    assert [event["index"] for event in events] == list(range(len(events)))
    assert final_units == round(game.book.payout_multiplier * scale)
    assert final_units == round(game.book.basegame_wins * scale)
    assert final_units == round(game.win_manager.running_bet_win * scale)
    assert final_units == round(game.win_manager.basegame_wins * scale)
    assert final_units == game.book.to_json()["payoutMultiplier"]
    assert game.book.freegame_wins == 0
    assert game.win_manager.freegame_wins == 0
    assert game.library[game.sim + 1] == game.book.to_json()


def test_public_surface_and_frozen_validator():
    assert last_shift.__all__ == ["GameConfig", "GameState"]
    assert not hasattr(last_shift, "LastShiftStateMachine")
    assert not hasattr(last_shift, "RoundState")
    assert issubclass(GameState, GeneralGameState)
    with pytest.raises(ImportError):
        exec("from games.last_shift import LastShiftStateMachine", {})
    with pytest.raises(ImportError):
        exec("from games.last_shift import RoundState", {})
    assert hashlib.sha256(
        inspect.getsource(validate_contract).encode()
    ).hexdigest() == FROZEN_VALIDATE_CONTRACT_HASH


def test_contract_proof_config_is_exact_and_not_publishable():
    config = GameConfig()
    assert config.contract_proof_only is True
    assert len(config.bet_modes) == 1
    mode = config.bet_modes[0]
    assert mode.get_name() == "contract_proof"
    assert mode.get_cost() == 1.0
    assert mode.get_rtp() == 0.0
    assert mode.get_wincap() == config.wincap
    assert mode.get_auto_close_disabled() is False
    assert mode.get_feature() is False
    assert mode.get_buybonus() is False

    distributions = mode.get_distributions()
    assert [item.get_criteria() for item in distributions] == [
        "contract_loss",
        "contract_departure",
    ]
    for distribution in distributions:
        assert distribution.get_fixed_amt() == 1
        assert distribution.get_quota() is None
        assert distribution.get_required_distribution_conditions() == []
        assert distribution._conditions == {
            "force_wincap": False,
            "force_freegame": False,
        }
        assert "reel_weights" not in distribution._conditions

    with pytest.raises(RuntimeError, match="not publishable"):
        config.assert_publishable_config()


def test_release_and_default_output_guards_fail_before_mutation(tmp_path):
    from src.state.run_sims import create_books

    called = False

    def downstream(*args, **kwargs):
        nonlocal called
        called = True

    proof_config = GameConfig()
    marker_before = proof_config.contract_proof_only
    try:
        proof_config.contract_proof_only = False
        with pytest.raises(RuntimeError, match="not publishable"):
            build_release(
                config=proof_config,
                publish_action=downstream,
                num_sim_args={"contract_proof": 2},
            )
    finally:
        proof_config.contract_proof_only = marker_before
    assert called is False

    repo_root = Path(__file__).resolve().parents[1]
    real_library = repo_root / "games" / "last_shift" / "library"
    files_before = sorted(
        str(path.relative_to(real_library))
        for path in real_library.rglob("*")
        if path.is_file()
    ) if real_library.exists() else []
    default_config = GameConfig()
    default_config.game_id = "last_shift"
    default_config.construct_paths()
    game = GameState(default_config)
    game.betmode = "contract_proof"
    game.criteria = "contract_loss"
    book_before = deepcopy(game.book.to_json())
    library_before = deepcopy(game.library)
    payouts_before = list(game._payout_ints)

    with pytest.raises(RuntimeError, match="canonical Last Shift library"):
        game.run_spin(0, 1)
    assert game.book.to_json() == book_before
    assert game.library == library_before
    assert game._payout_ints == payouts_before

    with pytest.raises(RuntimeError, match="canonical Last Shift library"):
        create_books(
            game,
            default_config,
            {"contract_proof": 2},
            batch_size=2,
            threads=1,
            compress=False,
            profiling=False,
        )

    split_config = GameConfig()
    split_game = GameState(split_config)
    split_game.output_files.library_path = str(tmp_path / "decoy_library")
    split_game.betmode = "contract_proof"
    split_game.criteria = "contract_departure"
    split_book_before = deepcopy(split_game.book.to_json())
    split_library_before = deepcopy(split_game.library)
    split_payouts_before = list(split_game._payout_ints)

    with pytest.raises(RuntimeError, match="canonical Last Shift library"):
        split_game.run_spin(0, 2)
    assert split_game.book.to_json() == split_book_before
    assert split_game.library == split_library_before
    assert split_game._payout_ints == split_payouts_before

    relative_config, relative_game = make_game(tmp_path)
    relative_game.output_files.book_path = "games/last_shift/library/books"
    relative_game.criteria = "contract_loss"
    relative_book_before = deepcopy(relative_game.book.to_json())
    relative_library_before = deepcopy(relative_game.library)
    relative_payouts_before = list(relative_game._payout_ints)
    relative_target = (
        repo_root / "games" / "last_shift" / "library" / "books"
    ).resolve()

    with pytest.raises(RuntimeError) as relative_error:
        relative_game.run_spin(0, 3)
    assert str(relative_target) in str(relative_error.value)
    assert relative_game.book.to_json() == relative_book_before
    assert relative_game.library == relative_library_before
    assert relative_game._payout_ints == relative_payouts_before
    assert relative_config.game_id == str(
        tmp_path / "last_shift_contract_proof_direct"
    )

    files_after = sorted(
        str(path.relative_to(real_library))
        for path in real_library.rglob("*")
        if path.is_file()
    ) if real_library.exists() else []
    assert files_after == files_before


def test_loss_book_uses_sdk_accounting_and_independent_validation(tmp_path):
    config, game = make_game(tmp_path)
    assert isinstance(game, GameState)
    book = run_criterion(game, "contract_loss")

    assert [event["type"] for event in book["events"]] == [
        "round_start",
        "board_reveal",
        "no_win",
        "round_complete",
    ]
    assert book["payoutMultiplier"] == 0
    assert all(event["type"] != "win_result" for event in book["events"])
    validate_contract(book["events"], config.wincap_units)
    assert_reconciled(config, game, book)


def test_departure_book_uses_sdk_accounting_and_independent_validation(tmp_path):
    config, game = make_game(tmp_path)
    book = run_criterion(game, "contract_departure")
    types = [event["type"] for event in book["events"]]

    assert book["payoutMultiplier"] > 0
    for required in (
        "columns_load",
        "cascade_refill",
        "board_modifier",
        "departure_prepare",
        "departure_resolve",
    ):
        assert required in types
    validate_contract(book["events"], config.wincap_units)
    assert_reconciled(config, game, book)


def test_reproducibility_and_cross_run_reset_on_one_instance(tmp_path):
    config, game = make_game(tmp_path)
    first_departure = run_criterion(game, "contract_departure", sim=7, seed=9001)
    second_departure = run_criterion(game, "contract_departure", sim=7, seed=9001)
    assert second_departure == first_departure

    loss = run_criterion(game, "contract_loss", sim=8, seed=9002)
    assert [event["type"] for event in loss["events"]] == [
        "round_start",
        "board_reveal",
        "no_win",
        "round_complete",
    ]
    assert loss["payoutMultiplier"] == 0
    assert game.book.payout_multiplier == 0
    assert game.book.basegame_wins == 0
    assert game.book.freegame_wins == 0
    assert game.final_win == 0
    assert game.wincap_triggered is False
    assert game.triggered_freegame is False
    assert game.win_manager.running_bet_win == 0
    assert game.win_manager.basegame_wins == 0
    assert game.win_manager.freegame_wins == 0
    assert game.win_manager.spin_win == 0
    assert game.win_manager.tumble_win == 0
    terminal = loss["events"][-1]
    assert terminal["finalLevels"] == [0, 0, 0, 0, 0, 0]
    assert terminal["finalDepartures"] == 0
    assert terminal["finalMode"] == config.basegame_type
    assert terminal["finalStage"] == "yard"
    assert terminal["finalFreeSpinsRemaining"] == 0
    assert terminal["capped"] is False
    assert game._last_shift_state.column_levels == (0, 0, 0, 0, 0, 0)
    assert game._last_shift_state.departures == 0
    assert game._last_shift_state.stage == "yard"

    third_departure = run_criterion(game, "contract_departure", sim=7, seed=9001)
    assert third_departure == first_departure
    validate_contract(third_departure["events"], config.wincap_units)


def test_unknown_criterion_fails_before_library_publication(tmp_path):
    _, game = make_game(tmp_path)
    library_before = deepcopy(game.library)
    payout_before = list(game._payout_ints)
    game.criteria = "unknown"
    with pytest.raises(ValueError, match="unsupported contract_proof criterion"):
        game.run_spin(0, 1)
    assert game.library == library_before
    assert game._payout_ints == payout_before


def test_cumulative_sdk_accounting_and_mutated_copy_rejection(tmp_path):
    config, game = make_game(tmp_path)
    loss = run_criterion(game, "contract_loss", sim=0, seed=1)
    departure = run_criterion(game, "contract_departure", sim=1, seed=2)
    expected_multiplier = sum(
        book["events"][-1]["finalPayoutUnits"] / config.payout_scale
        for book in (loss, departure)
    )
    assert game.win_manager.total_cumulative_wins == expected_multiplier
    assert game.win_manager.cumulative_base_wins == expected_multiplier
    assert game.win_manager.cumulative_free_wins == 0
    assert sorted(game.library) == [1, 2]

    original = deepcopy(departure["events"])
    mutated = [
        deepcopy(event)
        for event in original
        if event["type"] != "departure_resolve"
    ]
    for index, event in enumerate(mutated):
        event["index"] = index
    with pytest.raises(ValueError):
        validate_contract(mutated, config.wincap_units)
    assert departure["events"] == original
    validate_contract(departure["events"], config.wincap_units)


def test_create_books_pipeline_isolated_in_subprocess(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    real_library = repo_root / "games" / "last_shift" / "library"
    before = sorted(
        str(path.relative_to(real_library))
        for path in real_library.rglob("*")
        if path.is_file()
    ) if real_library.exists() else []
    isolated_game = tmp_path / "last_shift_contract_proof"
    completed = subprocess.run(
        [
            str(repo_root / "env" / "bin" / "python"),
            str(Path(__file__).resolve()),
            "--create-books-probe",
            str(isolated_game),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        text=True,
        capture_output=True,
        check=True,
    )
    result_line = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("T4C1G_RESULT=")
    )
    result = json.loads(result_line.removeprefix("T4C1G_RESULT="))
    assert result["criteria"] == ["contract_departure", "contract_loss"]
    assert sorted(result["payouts"]) == [0, 4960]
    assert result["library_keys"] == [1, 2]
    assert result["cumulative"] == 49.6

    output_root = isolated_game / "library"
    files = [path for path in output_root.rglob("*") if path.is_file()]
    assert files
    assert all(path.is_relative_to(isolated_game) for path in files)
    assert (output_root / "books" / "books_contract_proof.json").is_file()
    after = sorted(
        str(path.relative_to(real_library))
        for path in real_library.rglob("*")
        if path.is_file()
    ) if real_library.exists() else []
    assert after == before


def _run_create_books_probe(isolated_game):
    from src.state.run_sims import create_books

    config = GameConfig()
    config.game_id = str(isolated_game)
    config.construct_paths()
    game = GameState(config)
    create_books(
        game,
        config,
        {"contract_proof": 2},
        batch_size=2,
        threads=1,
        compress=False,
        profiling=False,
    )
    books = [game.library[key] for key in sorted(game.library)]
    for book in books:
        validate_contract(book["events"], config.wincap_units)
    print("T4C1G_RESULT=" + json.dumps({
        "criteria": sorted(book["criteria"] for book in books),
        "payouts": [book["payoutMultiplier"] for book in books],
        "library_keys": sorted(game.library),
        "cumulative": game.win_manager.total_cumulative_wins,
    }, sort_keys=True))


if __name__ == "__main__" and len(sys.argv) == 3:
    if sys.argv[1] != "--create-books-probe":
        raise SystemExit(f"unknown command: {sys.argv[1]}")
    _run_create_books_probe(Path(sys.argv[2]))
