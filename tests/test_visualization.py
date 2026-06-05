import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from tetris_rl.env import TetrisEnv
from tetris_rl.visualization import FallingPiece, PygameViewer, ViewerEvent


@pytest.fixture
def viewer():
    instance = PygameViewer(cell_size=12, panel_width=220)
    yield instance
    instance.close()


def test_viewer_draws_environment(viewer):
    env = TetrisEnv(seed=42)
    env.current_piece = "T"
    env.next_piece = "L"
    env._valid_actions = None
    env.step(0)

    viewer.draw(
        env,
        agent_name="RandomAgent",
        speed=8.0,
        total_reward=-2.0,
    )

    assert viewer.surface.get_size() == viewer.window_size
    assert viewer.surface.get_at((0, 0))[:3] == viewer.theme.background
    piece_pixel = (
        viewer.margin + viewer.cell_size + viewer.cell_size // 2,
        viewer.margin + (env.height - 1) * viewer.cell_size + viewer.cell_size // 2,
    )
    assert viewer.surface.get_at(piece_pixel)[:3] == viewer.theme.piece_colors[2]


def test_viewer_translates_keyboard_events(viewer):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

    events = viewer.poll_events()

    assert events == [
        ViewerEvent.TOGGLE_PAUSE,
        ViewerEvent.SPEED_UP,
        ViewerEvent.RESTART,
    ]


def test_falling_piece_advances_and_lands():
    env = TetrisEnv(seed=42)
    action = env.get_valid_actions()[0]
    piece = FallingPiece.from_action(env.current_piece, action)

    assert not piece.advance(100, rows_per_second=10)
    assert piece.y == 1
    assert piece.advance(10_000, rows_per_second=10)
    assert piece.y == action.y


def test_falling_piece_rotates_before_falling():
    env = TetrisEnv(seed=42)
    env.current_piece = "T"
    env._valid_actions = None
    action = next(action for action in env.get_valid_actions() if action.rotation == 2)
    piece = FallingPiece.from_action(env.current_piece, action)

    assert piece.is_rotating
    assert not piece.advance(140, rows_per_second=10)
    assert piece.displayed_rotation == 1
    assert piece.y == 0
    assert not piece.advance(140, rows_per_second=10)
    assert piece.displayed_rotation == 2
    assert not piece.is_rotating
    assert piece.y == 0
    assert not piece.advance(100, rows_per_second=10)
    assert piece.y == 1


def test_falling_piece_rejects_invalid_animation_speed():
    env = TetrisEnv(seed=42)
    piece = FallingPiece.from_action(env.current_piece, env.get_valid_actions()[0])

    with pytest.raises(ValueError, match="rows_per_second"):
        piece.advance(10, rows_per_second=0)
    with pytest.raises(ValueError, match="rotation_interval_ms"):
        piece.advance(10, rows_per_second=10, rotation_interval_ms=0)


def test_viewer_draws_falling_piece_overlay(viewer):
    env = TetrisEnv(seed=42)
    env.current_piece = "O"
    env._valid_actions = None
    action = env.get_valid_actions()[0]
    piece = FallingPiece.from_action(env.current_piece, action)

    viewer.draw(
        env,
        agent_name="HeuristicAgent",
        speed=30.0,
        total_reward=0.0,
        falling_piece=piece,
    )

    overlay_pixel = (
        viewer.margin + viewer.cell_size // 2,
        viewer.margin + viewer.cell_size // 2,
    )
    assert viewer.surface.get_at(overlay_pixel)[:3] == viewer.theme.piece_colors[1]


def test_viewer_rejects_tiny_cells():
    with pytest.raises(ValueError, match="cell_size"):
        PygameViewer(cell_size=4)
