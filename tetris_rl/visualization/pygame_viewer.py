"""Pygame renderer for candidate-placement Tetris episodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pygame

from tetris_rl.env import Action, TetrisEnv
from tetris_rl.env.pieces import PIECES, PIECE_IDS, rotation_size


class ViewerEvent(Enum):
    NONE = auto()
    QUIT = auto()
    TOGGLE_PAUSE = auto()
    RESTART = auto()
    SPEED_UP = auto()
    SPEED_DOWN = auto()


@dataclass(slots=True)
class FallingPiece:
    """Visual-only rotation and falling state for a selected placement."""

    piece_name: str
    target_rotation: int
    x: int
    target_y: int
    y: float = 0.0
    displayed_rotation: int = 0
    rotation_elapsed_ms: int = 0

    @classmethod
    def from_action(cls, piece_name: str, action: Action) -> FallingPiece:
        return cls(piece_name, action.rotation, action.x, action.y)

    @property
    def is_rotating(self) -> bool:
        return self.displayed_rotation != self.target_rotation

    def advance(
        self,
        elapsed_ms: int,
        rows_per_second: float,
        rotation_interval_ms: int = 140,
    ) -> bool:
        """Rotate first, then advance toward the final row."""
        if rows_per_second <= 0:
            raise ValueError("rows_per_second must be greater than zero")
        if rotation_interval_ms <= 0:
            raise ValueError("rotation_interval_ms must be greater than zero")

        if self.is_rotating:
            self.rotation_elapsed_ms += elapsed_ms
            rotations = PIECES[self.piece_name]
            while (
                self.rotation_elapsed_ms >= rotation_interval_ms
                and self.is_rotating
            ):
                self.rotation_elapsed_ms -= rotation_interval_ms
                self.displayed_rotation = (self.displayed_rotation + 1) % len(rotations)
            return False

        self.y = min(self.target_y, self.y + elapsed_ms * rows_per_second / 1_000)
        return self.y >= self.target_y


@dataclass(frozen=True, slots=True)
class ViewerTheme:
    background: tuple[int, int, int] = (12, 17, 28)
    panel: tuple[int, int, int] = (22, 30, 46)
    empty_cell: tuple[int, int, int] = (28, 38, 56)
    grid: tuple[int, int, int] = (43, 57, 79)
    primary_text: tuple[int, int, int] = (235, 241, 250)
    secondary_text: tuple[int, int, int] = (145, 164, 190)
    accent: tuple[int, int, int] = (250, 204, 21)
    piece_colors: tuple[tuple[int, int, int], ...] = (
        (67, 190, 235),   # I
        (250, 204, 21),   # O
        (168, 85, 247),   # T
        (74, 222, 128),   # S
        (248, 113, 113),  # Z
        (96, 165, 250),   # J
        (251, 146, 60),   # L
    )


class PygameViewer:
    """Render an environment and translate keyboard input into viewer events."""

    def __init__(
        self,
        board_width: int = 10,
        board_height: int = 20,
        *,
        cell_size: int = 30,
        panel_width: int = 280,
        caption: str = "Tetris RL",
        theme: ViewerTheme | None = None,
    ):
        if cell_size < 8:
            raise ValueError("cell_size must be at least 8")
        self.board_width = board_width
        self.board_height = board_height
        self.cell_size = cell_size
        self.panel_width = panel_width
        self.margin = 24
        self.theme = theme or ViewerTheme()
        self.window_size = (
            board_width * cell_size + panel_width + self.margin * 3,
            board_height * cell_size + self.margin * 2,
        )

        pygame.init()
        pygame.display.set_caption(caption)
        self.surface = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 38)
        self.body_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

    def poll_events(self) -> list[ViewerEvent]:
        """Convert relevant Pygame events into viewer-level commands."""
        translated: list[ViewerEvent] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                translated.append(ViewerEvent.QUIT)
            elif event.type == pygame.KEYDOWN:
                mapping = {
                    pygame.K_ESCAPE: ViewerEvent.QUIT,
                    pygame.K_SPACE: ViewerEvent.TOGGLE_PAUSE,
                    pygame.K_r: ViewerEvent.RESTART,
                    pygame.K_UP: ViewerEvent.SPEED_UP,
                    pygame.K_RIGHT: ViewerEvent.SPEED_UP,
                    pygame.K_DOWN: ViewerEvent.SPEED_DOWN,
                    pygame.K_LEFT: ViewerEvent.SPEED_DOWN,
                }
                translated.append(mapping.get(event.key, ViewerEvent.NONE))
        return translated

    def draw(
        self,
        env: TetrisEnv,
        *,
        agent_name: str,
        speed: float,
        total_reward: float,
        paused: bool = False,
        falling_piece: FallingPiece | None = None,
    ) -> None:
        """Draw the latest locked board and episode statistics."""
        self.surface.fill(self.theme.background)
        self._draw_board(env)
        if falling_piece is not None:
            self._draw_falling_piece(falling_piece)
        self._draw_panel(env, agent_name, speed, total_reward, paused)
        pygame.display.flip()

    def tick(self, fps: int = 60) -> int:
        """Limit the render loop and return elapsed milliseconds."""
        return self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

    def _draw_board(self, env: TetrisEnv) -> None:
        board_rect = pygame.Rect(
            self.margin,
            self.margin,
            self.board_width * self.cell_size,
            self.board_height * self.cell_size,
        )
        pygame.draw.rect(self.surface, self.theme.panel, board_rect, border_radius=6)
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(
                    self.margin + x * self.cell_size,
                    self.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                piece_id = int(env.board[y, x])
                color = (
                    self.theme.piece_colors[piece_id - 1]
                    if piece_id
                    else self.theme.empty_cell
                )
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, self.theme.grid, rect, width=1)

    def _draw_falling_piece(self, piece: FallingPiece) -> None:
        rotation = PIECES[piece.piece_name][piece.displayed_rotation]
        color = self.theme.piece_colors[PIECE_IDS[piece.piece_name] - 1]
        board_y = int(piece.y)
        width, _ = rotation_size(rotation)
        board_x = min(piece.x, self.board_width - width)
        for dx, dy in rotation:
            x, y = board_x + dx, board_y + dy
            if not (0 <= x < self.board_width and 0 <= y < self.board_height):
                continue
            rect = pygame.Rect(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.surface, color, rect, border_radius=3)
            pygame.draw.rect(self.surface, self.theme.grid, rect, width=1)

    def _draw_panel(
        self,
        env: TetrisEnv,
        agent_name: str,
        speed: float,
        total_reward: float,
        paused: bool,
    ) -> None:
        panel_x = self.margin * 2 + self.board_width * self.cell_size
        panel_rect = pygame.Rect(
            panel_x,
            self.margin,
            self.panel_width,
            self.board_height * self.cell_size,
        )
        pygame.draw.rect(self.surface, self.theme.panel, panel_rect, border_radius=10)

        y = self.margin + 26
        self._text("TETRIS RL", panel_x + 22, y, self.title_font, self.theme.accent)
        y += 48
        self._text(agent_name, panel_x + 22, y, self.body_font, self.theme.primary_text)
        y += 42

        self._text("Current", panel_x + 22, y, self.small_font, self.theme.secondary_text)
        self._text("Next", panel_x + 142, y, self.small_font, self.theme.secondary_text)
        self._draw_piece_preview(env.current_piece, panel_x + 12, y + 24)
        self._draw_piece_preview(env.next_piece, panel_x + 132, y + 24)
        y += 104

        metrics = (
            ("Score", str(env.score)),
            ("Lines", str(env.lines_cleared)),
            ("Pieces", str(env.pieces_placed)),
            ("Reward", f"{total_reward:.1f}"),
            ("Fall speed", f"{speed:.1f} rows/s"),
        )
        for label, value in metrics:
            self._text(label, panel_x + 22, y, self.small_font, self.theme.secondary_text)
            self._text(value, panel_x + 22, y + 20, self.body_font, self.theme.primary_text)
            y += 52

        status = "PAUSED" if paused else ("GAME OVER" if env.done else "PLAYING")
        status_color = (
            self.theme.accent if paused or env.done else self.theme.piece_colors[0]
        )
        self._text(status, panel_x + 22, y, self.body_font, status_color)

        controls_y = panel_rect.bottom - 94
        controls = ("Space  pause", "R  restart", "Arrows  speed", "Esc  quit")
        for line in controls:
            self._text(
                line,
                panel_x + 22,
                controls_y,
                self.small_font,
                self.theme.secondary_text,
            )
            controls_y += 22

    def _draw_piece_preview(self, piece_name: str, x: int, y: int) -> None:
        rotation = PIECES[piece_name][0]
        width, height = rotation_size(rotation)
        preview_cell = 22
        preview_width = 5 * preview_cell
        origin_x = x + (preview_width - width * preview_cell) // 2
        origin_y = y + (3 * preview_cell - height * preview_cell) // 2
        color = self.theme.piece_colors[PIECE_IDS[piece_name] - 1]

        for dx, dy in rotation:
            rect = pygame.Rect(
                origin_x + dx * preview_cell,
                origin_y + dy * preview_cell,
                preview_cell,
                preview_cell,
            )
            pygame.draw.rect(self.surface, color, rect, border_radius=3)
            pygame.draw.rect(self.surface, self.theme.panel, rect, width=2, border_radius=3)

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        self.surface.blit(font.render(text, True, color), (x, y))
