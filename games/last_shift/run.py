"""Guarded Last Shift release entrypoint."""

from games.last_shift.game_config import GameConfig
from games.last_shift.gamestate import GameState
from src.state.run_sims import create_books


def build_release(
    *,
    config=None,
    publish_action=create_books,
    num_sim_args=None,
    batch_size=1,
    threads=1,
    compress=True,
    profiling=False,
):
    """Build publish artifacts only after the game config passes its guard."""
    config = config or GameConfig()
    config.assert_publishable_config()
    gamestate = GameState(config)
    return publish_action(
        gamestate,
        config,
        num_sim_args or {},
        batch_size,
        threads,
        compress,
        profiling,
    )


if __name__ == "__main__":
    build_release()
