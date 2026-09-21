"""Project 11: travel between cities on a cylindrical hex map with a barbarian.

Demo: python -m reinforcement_learning.projects.project011 --episodes 1000
Study: python -m reinforcement_learning.projects.project011 --study --episodes 100000
The demo uses the standard library; study charts require Matplotlib. Terrain
is fixed across episodes. See docs/projects/project11.md.
"""

import argparse
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from pathlib import Path
from statistics import mean

MAX_HEALTH = 100
MOVES_PER_TURN = 2


class Terrain(Enum):
    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAIN = "mountain"


def movement_cost(terrain: Terrain) -> int:
    """Cost to ENTER a tile. Both rough terrains exhaust a fresh two-move turn."""
    return 1 if terrain is Terrain.PLAINS else 2


class Action(IntEnum):
    EAST = 0
    NORTHEAST = 1
    NORTHWEST = 2
    WEST = 3
    SOUTHWEST = 4
    SOUTHEAST = 5
    REST = 6
    ATTACK = 7


MOVE_ACTIONS = tuple(Action(index) for index in range(6))


@dataclass(frozen=True, slots=True)
class Hex:
    """Canonical odd-row-offset coordinates; odd rows are shifted right."""

    col: int
    row: int

    def __post_init__(self) -> None:
        if type(self.col) is not int or type(self.row) is not int:
            raise ValueError("Hex coordinates must be integers")


@dataclass(frozen=True)
class HexMap:
    tiles: tuple[tuple[Terrain, ...], ...]
    city1: Hex
    city2: Hex
    _steps: tuple[tuple[Hex | None, ...], ...] = field(init=False, repr=False, compare=False)
    _adjacent: tuple[tuple[Hex, ...], ...] = field(init=False, repr=False, compare=False)
    _distances: dict[int, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Freeze even caller-provided mutable rows so reset cannot change terrain.
        object.__setattr__(self, "tiles", tuple(tuple(row) for row in self.tiles))
        if not self.tiles or self.width < 3 or any(len(row) != self.width for row in self.tiles):
            raise ValueError("Use a rectangular map at least three tiles wide and one tile tall")
        if any(not isinstance(tile, Terrain) for row in self.tiles for tile in row):
            raise ValueError("Every tile must be a Terrain")
        if not self.contains(self.city1) or not self.contains(self.city2) or self.city1 == self.city2:
            raise ValueError("Cities must be distinct tiles inside the map")
        steps: list[tuple[Hex | None, ...]] = []
        for row in range(self.height):
            shift = row % 2
            offsets = ((1, 0), (shift, -1), (shift - 1, -1), (-1, 0), (shift - 1, 1), (shift, 1))
            for col in range(self.width):
                steps.append(
                    tuple(
                        Hex((col + dx) % self.width, row + dy) if 0 <= row + dy < self.height else None
                        for dx, dy in offsets
                    )
                )
        object.__setattr__(self, "_steps", tuple(steps))
        object.__setattr__(self, "_adjacent", tuple(tuple(tile for tile in row if tile is not None) for row in steps))

    @property
    def width(self) -> int:
        return len(self.tiles[0])

    @property
    def height(self) -> int:
        return len(self.tiles)

    def contains(self, tile: Hex) -> bool:
        return 0 <= tile.col < self.width and 0 <= tile.row < self.height

    @classmethod
    def generate(cls, seed: int = 42) -> HexMap:
        """Fixed 20x20 layout, 60% plains / 25% forest / 15% mountain draws."""
        # Seeded terrain simulation, not cryptography.
        rng = random.Random(seed)  # nosec B311
        rows = [rng.choices(tuple(Terrain), weights=(60, 25, 15), k=20) for _ in range(20)]
        city1, city2 = Hex(2, 10), Hex(12, 10)
        for city in (city1, city2):
            rows[city.row][city.col] = Terrain.PLAINS
        return cls(tuple(tuple(row) for row in rows), city1, city2)

    def neighbor(self, tile: Hex, action: Action) -> Hex | None:
        if not self.contains(tile) or action not in MOVE_ACTIONS:
            raise ValueError("A move requires a map tile and a movement action")
        return self._steps[tile.row * self.width + tile.col][action]

    def neighbors(self, tile: Hex) -> tuple[Hex, ...]:
        if not self.contains(tile):
            raise ValueError("Neighbors require a tile inside the map")
        return self._adjacent[tile.row * self.width + tile.col]

    def distance(self, left: Hex, right: Hex) -> int:
        """Shortest hex distance including the east/west seam, ignoring terrain."""
        if not self.contains(left) or not self.contains(right):
            raise ValueError("Distance requires tiles inside the map")
        indices = sorted((left.row * self.width + left.col, right.row * self.width + right.col))
        key = indices[0] * self.width * self.height + indices[1]
        if key in self._distances:
            return self._distances[key]
        left_q = left.col - (left.row - left.row % 2) // 2
        right_q = right.col - (right.row - right.row % 2) // 2
        dr = left.row - right.row
        distance = min(
            max(abs(dq), abs(dr), abs(dq + dr))
            for wrap in (-self.width, 0, self.width)
            for dq in (left_q - right_q + wrap,)
        )
        self._distances[key] = distance
        return distance


@dataclass(frozen=True)
class Rules:
    max_turns: int = 500
    heal_amount: int = 10
    city_damage: int = 30
    city_radius: int = 2

    def __post_init__(self) -> None:
        for name in ("max_turns", "heal_amount", "city_damage", "city_radius"):
            value = getattr(self, name)
            minimum = 1 if name in ("max_turns", "heal_amount") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class Outcome(Enum):
    RUNNING = "running"
    SUCCESS = "reached city 2"
    DEATH = "player died"
    TIMEOUT = "turn limit"


@dataclass(frozen=True, slots=True)
class GameState:
    player: Hex
    player_health: int
    barbarian: Hex | None
    barbarian_health: int
    moves_left: int = MOVES_PER_TURN
    turns_elapsed: int = 0
    outcome: Outcome = Outcome.RUNNING


@dataclass(frozen=True)
class StepResult:
    state: GameState
    reward: float
    events: tuple[str, ...] = ()

    @property
    def terminated(self) -> bool:
        return self.state.outcome in (Outcome.SUCCESS, Outcome.DEATH)

    @property
    def truncated(self) -> bool:
        return self.state.outcome is Outcome.TIMEOUT

    @property
    def done(self) -> bool:
        return self.state.outcome is not Outcome.RUNNING


def _check_health(value: int, *, allow_dead: bool = False) -> None:
    minimum = 0 if allow_dead else 1
    if type(value) is not int or not minimum <= value <= MAX_HEALTH:
        raise ValueError(f"Health must be an integer from {minimum} to {MAX_HEALTH}")


def combat_damage(attacker_health: int, defender_health: int) -> int:
    """Editable one-sided strike: round(30 * attacker HP / defender HP).

    Clamp to [1, defender HP]. Equal-health units deal 30 damage unless the
    defender has fewer than 30 HP. No terrain, defense, or retaliation bonuses.
    """
    _check_health(attacker_health)
    _check_health(defender_health)
    return min(defender_health, max(1, round(30 * attacker_health / defender_health)))


def barbarian_attack_probability(barbarian_health: int, player_health: int) -> float:
    _check_health(barbarian_health, allow_dead=True)
    _check_health(player_health)
    return 1.0 if barbarian_health > player_health else barbarian_health / MAX_HEALTH


def allowed_actions(board: HexMap, state: GameState) -> tuple[Action, ...]:
    """Exclude off-map movement, occupied destinations, and out-of-range attacks."""
    if state.outcome is not Outcome.RUNNING:
        return ()
    actions = [
        action
        for action in MOVE_ACTIONS
        if (target := board.neighbor(state.player, action)) is not None and target != state.barbarian
    ]
    actions.append(Action.REST)
    if state.barbarian is not None and board.distance(state.player, state.barbarian) == 1:
        actions.append(Action.ATTACK)
    return tuple(actions)


class HexEnvironment:
    """Player decisions, then one barbarian phase per completed player turn.

    Reset restores the player to city 1 and randomizes the barbarian, never the
    terrain. Dead units do not respawn until reset. Observations reveal both
    units; this initial project has no fog of war.
    """

    def __init__(
        self,
        board: HexMap | None = None,
        *,
        seed: int = 42,
        rules: Rules | None = None,
        damage_function: Callable[[int, int], int] = combat_damage,
    ) -> None:
        self.board = board if board is not None else HexMap.generate()
        self.rules = rules if rules is not None else Rules()
        self.damage_function = damage_function
        # Reproducible game simulation, not cryptography.
        self._rng = random.Random(seed)  # nosec B311
        self._state: GameState | None = None
        self._spawn_tiles = tuple(
            Hex(col, row)
            for row in range(self.board.height)
            for col in range(self.board.width)
            if Hex(col, row) != self.board.city1
        )

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("Call reset() before using the environment")
        return self._state

    def reset(
        self,
        *,
        barbarian_position: Hex | None = None,
        player_health: int = MAX_HEALTH,
        barbarian_health: int = MAX_HEALTH,
        seed: int | None = None,
    ) -> GameState:
        """Optional spawn/health overrides; a seed pairs episode starts across policies."""
        _check_health(player_health)
        _check_health(barbarian_health, allow_dead=True)
        if barbarian_position is not None and (
            not self.board.contains(barbarian_position) or barbarian_position == self.board.city1
        ):
            raise ValueError("The barbarian must spawn on a valid tile other than the occupied city 1")
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("Episode seed must be an integer")
            self._rng.seed(seed)
        barbarian = (barbarian_position or self._rng.choice(self._spawn_tiles)) if barbarian_health else None
        self._state = GameState(self.board.city1, player_health, barbarian, barbarian_health)
        return self._state

    def _damage(self, attacker: int, defender: int) -> int:
        damage = self.damage_function(attacker, defender)
        if type(damage) is not int or damage < 0:
            raise ValueError("damage_function must return a nonnegative integer")
        return min(defender, damage)

    def _roam(self, state: GameState) -> GameState:
        if state.barbarian is None:
            return state
        choices = (state.barbarian,) + tuple(
            tile for tile in self.board.neighbors(state.barbarian) if tile != state.player
        )
        return replace(state, barbarian=self._rng.choice(choices))

    def _in_attack_range(self, state: GameState) -> bool:
        return (
            state.player != self.board.city1
            and state.barbarian is not None
            and self.board.distance(state.player, state.barbarian) == 1
        )

    def _barbarian_turn(self, state: GameState, events: list[str]) -> GameState:
        if state.barbarian is None:
            return state
        # Adjacent units decide once before moving. Otherwise roam, then decide
        # once if now adjacent. A declined attack cannot get a second roll.
        moved = not self._in_attack_range(state)
        if moved:
            state = self._roam(state)
        if self._in_attack_range(state):
            chance = barbarian_attack_probability(state.barbarian_health, state.player_health)
            if self._rng.random() < chance:
                damage = self._damage(state.barbarian_health, state.player_health)
                state = replace(state, player_health=state.player_health - damage)
                events.append(f"Barbarian dealt {damage} damage")
            elif not moved:
                state = self._roam(state)
        # Fire happens at the END of the barbarian turn, even if its attack
        # killed the player. Overlapping city radii do not stack.
        if state.barbarian is not None and any(
            self.board.distance(state.barbarian, city) <= self.rules.city_radius
            for city in (self.board.city1, self.board.city2)
        ):
            damage = min(state.barbarian_health, self.rules.city_damage)
            health = state.barbarian_health - damage
            state = replace(state, barbarian_health=health, barbarian=state.barbarian if health else None)
            if damage:
                events.append(f"City defenses dealt {damage} damage")
        return state

    def step(self, action: Action) -> StepResult:
        state = self.state
        if state.outcome is not Outcome.RUNNING:
            raise RuntimeError("The episode is over; call reset()")
        if not isinstance(action, Action) or action not in allowed_actions(self.board, state):
            raise ValueError("Action is not allowed in this state")
        events: list[str] = []
        if action is Action.REST:
            health = min(MAX_HEALTH, state.player_health + self.rules.heal_amount)
            events.append(f"Rest healed {health - state.player_health} health")
            state = replace(state, player_health=health, moves_left=0)
        elif action is Action.ATTACK:
            damage = self._damage(state.player_health, state.barbarian_health)
            health = state.barbarian_health - damage
            state = replace(state, barbarian_health=health, barbarian=state.barbarian if health else None, moves_left=0)
            events.append(f"Player dealt {damage} damage")
        else:
            target = self.board.neighbor(state.player, action)
            if target is None:
                raise RuntimeError("Legal move had no destination")
            remaining = max(0, state.moves_left - movement_cost(self.board.tiles[target.row][target.col]))
            state = replace(state, player=target, moves_left=remaining)
            if target == self.board.city2:
                # Arrival ends immediately, without an enemy response; a partial
                # final turn still counts as one turn of travel.
                state = replace(state, outcome=Outcome.SUCCESS, moves_left=0, turns_elapsed=state.turns_elapsed + 1)
                self._state = state
                return StepResult(state, 99.0, ("Reached city 2",))
        reward = 0.0
        if state.moves_left == 0:
            state = self._barbarian_turn(state, events)
            elapsed = state.turns_elapsed + 1
            outcome = Outcome.RUNNING
            if state.player_health == 0:
                outcome = Outcome.DEATH
            elif elapsed >= self.rules.max_turns:
                outcome = Outcome.TIMEOUT
            state = replace(
                state,
                turns_elapsed=elapsed,
                outcome=outcome,
                moves_left=MOVES_PER_TURN if outcome is Outcome.RUNNING else 0,
            )
            reward = -1.0 if outcome is Outcome.RUNNING else -1001.0
        self._state = state
        return StepResult(state, reward, tuple(events))


# Nearby enemies retain exact coordinates/HP. Distant position is aggregated;
# all horizons above ten turns share a bucket. This is a baseline, not an exact
# Markov-state solver. Keep the full GameState available to alternative agents.
LearningKey = tuple[Hex, int, int, Hex | None, int, int]


def learning_key(board: HexMap, rules: Rules, state: GameState) -> LearningKey:
    """Shared state aggregation for the demo and episodic policy comparison."""
    nearby = state.barbarian
    if nearby is not None and board.distance(state.player, nearby) > 3:
        nearby = None
    return (
        state.player,
        state.moves_left,
        state.player_health,
        nearby,
        state.barbarian_health,
        min(10, rules.max_turns - state.turns_elapsed),
    )


class QLearningAgent:
    """Masked Q-learning baseline with local threat aggregation and zero initial Q.

    Gamma=1; fixed alpha. Potential shaping guides initial exploration without
    adding reward for healing or combat. No giant per-episode-time state table.
    """

    def __init__(
        self, board: HexMap, *, rules: Rules | None = None, seed: int = 43, epsilon: float = 0.1, alpha: float = 0.1
    ) -> None:
        if not 0 <= epsilon <= 1 or not 0 < alpha <= 1:
            raise ValueError("epsilon must be in [0, 1] and alpha in (0, 1]")
        self.board = board
        self.rules = rules if rules is not None else Rules()
        self.epsilon, self.alpha = epsilon, alpha
        self.q: dict[LearningKey, list[float]] = {}
        # Reproducible exploratory actions, not cryptography.
        self._rng = random.Random(seed)  # nosec B311

    def key(self, state: GameState) -> LearningKey:
        return learning_key(self.board, self.rules, state)

    def potential(self, state: GameState) -> float:
        if state.outcome is not Outcome.RUNNING:
            return 0.0
        return -self.board.distance(state.player, self.board.city2) / MOVES_PER_TURN

    def choose_action(self, state: GameState, *, explore: bool = True) -> Action:
        actions = allowed_actions(self.board, state)
        if not actions:
            raise ValueError("There are no actions after the episode ends")
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        values = self.q.get(self.key(state), [0.0] * len(Action))
        best = max(values[action] for action in actions)
        return self._rng.choice([action for action in actions if values[action] == best])

    def learn(self, state: GameState, action: Action, result: StepResult) -> None:
        if not isinstance(action, Action) or action not in allowed_actions(self.board, state):
            raise ValueError("Cannot learn an illegal action")
        if not math.isfinite(result.reward):
            raise ValueError("Reward must be finite")
        target = result.reward + self.potential(result.state) - self.potential(state)
        if not result.done:
            next_values = self.q.get(self.key(result.state), [0.0] * len(Action))
            target += max(next_values[candidate] for candidate in allowed_actions(self.board, result.state))
        values = self.q.setdefault(self.key(state), [0.0] * len(Action))
        values[action] += self.alpha * (target - values[action])


@dataclass(frozen=True)
class EpisodeResult:
    outcome: Outcome
    turns: int
    decisions: int
    reward: float


def run_episode(
    environment: HexEnvironment,
    agent: QLearningAgent,
    *,
    learn: bool = True,
    trace: list[GameState] | None = None,
) -> EpisodeResult:
    """Training or frozen greedy evaluation; trace optionally records decisions."""
    if agent.board != environment.board or agent.rules != environment.rules:
        raise ValueError("Agent and environment must use the same map and rules")
    state = environment.reset()
    if trace is not None:
        trace.append(state)
    total, decisions = 0.0, 0
    while True:
        action = agent.choose_action(state, explore=learn)
        result = environment.step(action)
        if learn:
            agent.learn(state, action, result)
        total += result.reward
        decisions += 1
        state = result.state
        if trace is not None:
            trace.append(state)
        if result.done:
            return EpisodeResult(state.outcome, state.turns_elapsed, decisions, total)


def save_svg(board: HexMap, state: GameState, path: Path, *, trail: Sequence[Hex] = ()) -> None:
    """Draw actual pointy-top hexagons with terrain, cities, health, and a trail."""
    size, margin = 20.0, 30.0
    tile_width = math.sqrt(3) * size
    width = 2 * margin + tile_width * (board.width + 0.5)
    height = 2 * margin + 1.5 * size * (board.height - 1) + 2 * size + 65
    colors = {Terrain.PLAINS: "#d6cf8e", Terrain.FOREST: "#527b48", Terrain.MOUNTAIN: "#969ca2"}
    symbols = {Terrain.PLAINS: "·", Terrain.FOREST: "F", Terrain.MOUNTAIN: "M"}
    visited = set(trail)

    def center(tile: Hex) -> tuple[float, float]:
        return margin + tile_width * (tile.col + 0.5 + 0.5 * (tile.row % 2)), margin + size + 1.5 * size * tile.row

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" role="img">',
        "<title>Project 11 — cylindrical hex world</title>",
        "<desc>North and south are bounded. East and west wrap. Plains cost one move; forest and mountain cost two.</desc>",
        '<rect width="100%" height="100%" fill="#f5efdc"/>',
        '<g font-family="sans-serif" text-anchor="middle" font-size="11">',
    ]
    for row, tiles in enumerate(board.tiles):
        for col, terrain in enumerate(tiles):
            tile = Hex(col, row)
            x, y = center(tile)
            points = " ".join(
                f"{x + size * math.cos(math.radians(30 + 60 * corner)):.1f},"
                f"{y + size * math.sin(math.radians(30 + 60 * corner)):.1f}"
                for corner in range(6)
            )
            border = "#ffcd38" if tile in visited else "#f5efdc"
            lines.append(f'<polygon points="{points}" fill="{colors[terrain]}" stroke="{border}" stroke-width="2"/>')
            label = "C1" if tile == board.city1 else "C2" if tile == board.city2 else symbols[terrain]
            color = "#162a42" if tile in (board.city1, board.city2) else "#233222"
            label_y = y - 9 if tile in (board.city1, board.city2) else y + 4
            lines.append(f'<text x="{x:.1f}" y="{label_y:.1f}" fill="{color}">{label}</text>')
    for unit_tile, label, health, color in (
        (state.player, "P", state.player_health, "#215ca0"),
        (state.barbarian, "B", state.barbarian_health, "#b33530"),
    ):
        if unit_tile is not None:
            x, y = center(unit_tile)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>')
            lines.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" fill="white" font-size="9">{label}</text>')
            lines.append(f'<text x="{x:.1f}" y="{y + 17:.1f}" fill="{color}" font-size="9">{health} HP</text>')
    lines.extend(
        (
            f'<text x="{width / 2:.1f}" y="{height - 40:.1f}">C1 start · C2 goal · P player · B barbarian · yellow trail</text>',
            f'<text x="{width / 2:.1f}" y="{height - 23:.1f}">Plains (·): 1 move · Forest (F) / Mountain (M): 2 moves · E/W wrap</text>',
            f'<text x="{width / 2:.1f}" y="{height - 7:.1f}">Turns: {state.turns_elapsed} · {state.outcome.value}</text>',
            "</g></svg>",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(label: str, episodes: Sequence[EpisodeResult]) -> list[str]:
    successes = [episode for episode in episodes if episode.outcome is Outcome.SUCCESS]
    return [
        f"## {label} ({len(episodes)} episodes)",
        f"- Successes: {len(successes)}",
        f"- Deaths: {sum(episode.outcome is Outcome.DEATH for episode in episodes)}",
        f"- Timeouts: {sum(episode.outcome is Outcome.TIMEOUT for episode in episodes)}",
        (
            f"- Mean successful travel time: {mean(episode.turns for episode in successes):.2f} turns"
            if successes
            else "- Mean successful travel time: n/a"
        ),
        (
            f"- Mean unshaped return: {mean(episode.reward for episode in episodes):.2f}"
            if episodes
            else "- Mean return: n/a"
        ),
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, help="Training episodes (default: 1,000 demo; 100,000 per study configuration/run)"
    )
    parser.add_argument("--study", action="store_true", help="Compare four episodic methods and their parameters")
    parser.add_argument("--runs", type=int, default=1, help="Independent study runs per configuration")
    parser.add_argument("--workers", type=int, default=1, help="Study worker processes")
    parser.add_argument("--parameters", type=float, nargs="+", help="Optional study parameter grid")
    parser.add_argument("--block-size", type=int, default=1000, help="Episodes per saved learning-curve block")
    parser.add_argument("--evaluate", type=int, default=100, help="Separate greedy evaluation episodes (demo only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.1, help="Demo exploration rate; the study uses --parameters")
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Demo Q-learning step size; the study uses --parameters"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/projects/assets/project_11"))
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 100000 if args.study else 1000
    if args.study:
        from reinforcement_learning.projects.project011_study import PARAMETERS, run_study

        try:
            run_study(
                output_dir=args.output_dir / "study",
                episodes=args.episodes,
                runs=args.runs,
                seed=args.seed,
                map_seed=args.map_seed,
                max_turns=args.max_turns,
                workers=args.workers,
                parameters=tuple(args.parameters) if args.parameters is not None else PARAMETERS,
                block_size=args.block_size,
            )
        except ValueError as error:
            parser.error(str(error))
        return
    if args.episodes < 0 or args.evaluate < 1:
        parser.error("--episodes must be nonnegative and --evaluate must be positive")
    board = HexMap.generate(args.map_seed)
    try:
        rules = Rules(max_turns=args.max_turns)
        agent = QLearningAgent(board, rules=rules, seed=args.seed + 1, epsilon=args.epsilon, alpha=args.alpha)
    except ValueError as error:
        parser.error(str(error))
    environment = HexEnvironment(board, rules=rules, seed=args.seed)
    training: list[EpisodeResult] = []
    for index in range(args.episodes):
        training.append(run_episode(environment, agent))
        if (index + 1) % 100 == 0:
            print(f"Training episodes: {index + 1:,}; learned keys: {len(agent.q):,}", flush=True)
    evaluation_environment = HexEnvironment(board, rules=rules, seed=args.seed + 2)
    trace: list[GameState] = []
    evaluation = [run_episode(evaluation_environment, agent, learn=False, trace=trace)]
    evaluation.extend(run_episode(evaluation_environment, agent, learn=False) for _ in range(args.evaluate - 1))
    output: Path = args.output_dir
    save_svg(board, trace[0], output / "map.svg")
    save_svg(board, trace[-1], output / "episode.svg", trail=[state.player for state in trace])
    lines = [
        "# Project 11 — Hex-world travel",
        "",
        f"Map seed: {args.map_seed}; unit seed: {args.seed}; turn limit: {rules.max_turns}.",
        f"Training: epsilon={agent.epsilon:g}, alpha={agent.alpha:g}, gamma=1; zero initial Q; distance potential shaping.",
        f"Learned aggregate state keys: {len(agent.q)}. Evaluation is greedy with frozen estimates, not proof of optimality.",
        "",
        *_summary("Training", training),
        *_summary("Evaluation", evaluation),
        "The SVGs show the start and trail of the first evaluation episode only.",
        "No model weights, episode dataset, or large CSV are written.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {output / 'map.svg'}, {output / 'episode.svg'}, and {output / 'summary.md'}")


if __name__ == "__main__":
    main()
