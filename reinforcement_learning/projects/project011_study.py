"""Episode-level parameter study for Project 11, invoked with project011 --study.

Action-value methods use every-visit Monte Carlo returns; the gradient method
is episodic REINFORCE, not an immediate-reward bandit on movement directions.
Completed configuration/run checkpoints contain summaries only, never Q tables.
"""

import csv
import hashlib
import json
import math
import os
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import StringIO
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal

from reinforcement_learning.projects import project011 as game

Method = Literal["epsilon_greedy", "gradient", "optimistic", "ucb"]
METHODS: tuple[Method, ...] = ("epsilon_greedy", "gradient", "optimistic", "ucb")
PARAMETERS = tuple(2.0**exponent for exponent in range(-7, 3))
REWARD_SCALE = 1000.0
NAMES = {
    "epsilon_greedy": "Epsilon-greedy",
    "gradient": "Gradient bandit → episodic REINFORCE",
    "optimistic": "Greedy, optimistic initialization",
    "ucb": "State-local UCB",
}
LABELS = {
    "epsilon_greedy": r"$\varepsilon$-greedy ($\varepsilon$)",
    "gradient": r"Episodic gradient ($\alpha$)",
    "optimistic": r"Greedy, optimistic ($Q_0$; step size $0.1$)",
    "ucb": r"State-local UCB ($c$)",
}
COLORS = {"epsilon_greedy": "tab:red", "gradient": "tab:green", "optimistic": "tab:orange", "ucb": "tab:blue"}


@dataclass(frozen=True, slots=True)
class Decision:
    key: game.LearningKey
    actions: tuple[game.Action, ...]
    action: game.Action
    probabilities: tuple[float, ...] = ()  # Legal-action order; policy at sampling time.


@dataclass(frozen=True, slots=True)
class Transition:
    decision: Decision
    reward: float  # Original environment reward, never a normalized plot score.


class EpisodicAgent:
    """One fresh state-conditioned learner for one method, parameter, and run.

    Values/preferences remain fixed during each episode. UCB selection counts
    advance at each decision; Monte Carlo update counts advance at each update.
    Those counts must be distinct when a state-action repeats within an episode.
    """

    def __init__(self, board: game.HexMap, rules: game.Rules, method: Method, parameter: float, *, seed: int) -> None:
        if method not in METHODS or not math.isfinite(parameter) or parameter <= 0:
            raise ValueError("Choose a supported method and a positive finite parameter")
        if method in ("epsilon_greedy", "gradient") and parameter > 1:
            raise ValueError("Epsilon and gradient alpha are capped at one in this study")
        self.board, self.rules = board, rules
        self.method, self.parameter = method, parameter
        # Reproducible policy simulation, not cryptography.
        self.rng = random.Random(seed)  # nosec B311
        self.q: dict[game.LearningKey, list[float]] = {}
        self.preferences: dict[game.LearningKey, list[float]] = {}
        self.counts: dict[game.LearningKey, list[int]] = {}
        self.updates: dict[game.LearningKey, list[int]] = {}
        self.baseline: dict[game.LearningKey, float] = {}
        self.baseline_counts: dict[game.LearningKey, int] = {}

    def probabilities(self, key: game.LearningKey, actions: tuple[game.Action, ...]) -> tuple[float, ...]:
        if not actions:
            raise ValueError("Softmax needs at least one legal action")
        preferences = self.preferences.setdefault(key, [0.0] * len(game.Action))
        largest = max(preferences[action] for action in actions)
        weights = [math.exp(preferences[action] - largest) for action in actions]
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def decide(self, state: game.GameState) -> Decision:
        actions = game.allowed_actions(self.board, state)
        if not actions:
            raise ValueError("Cannot act after termination")
        key = game.learning_key(self.board, self.rules, state)
        if self.method == "gradient":
            probabilities = self.probabilities(key, actions)
            action = self.rng.choices(actions, weights=probabilities, k=1)[0]
            return Decision(key, actions, action, probabilities)
        initial = self.parameter if self.method == "optimistic" else 0.0
        values = self.q.setdefault(key, [initial] * len(game.Action))
        if self.method == "epsilon_greedy" and self.rng.random() < self.parameter:
            action = self.rng.choice(actions)
        elif self.method == "ucb":
            counts = self.counts.setdefault(key, [0] * len(game.Action))
            untried = [candidate for candidate in actions if counts[candidate] == 0]
            if untried:
                action = self.rng.choice(untried)
            else:
                log_visits = math.log(sum(counts) + 1)
                scores = {
                    candidate: values[candidate] + self.parameter * math.sqrt(log_visits / counts[candidate])
                    for candidate in actions
                }
                best = max(scores.values())
                action = self.rng.choice([candidate for candidate in actions if scores[candidate] == best])
            counts[action] += 1
        else:
            best = max(values[candidate] for candidate in actions)
            action = self.rng.choice([candidate for candidate in actions if values[candidate] == best])
        return Decision(key, actions, action)

    def update_episode(self, trajectory: Sequence[Transition]) -> None:
        """Use undiscounted returns-to-go, scaled identically for all four methods.

        REINFORCE sums all decision gradients, using the stored behavior policy
        and baselines from BEFORE this episode, including for repeated states.
        """
        if not trajectory:
            raise ValueError("An episode must have at least one decision")
        if any(
            not math.isfinite(step.reward) or step.decision.action not in step.decision.actions for step in trajectory
        ):
            raise ValueError("Episode rewards must be finite and actions legal")
        returns = [0.0] * len(trajectory)
        total = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            total += trajectory[index].reward / REWARD_SCALE
            returns[index] = total
        if self.method == "gradient":
            old_baselines = {step.decision.key: self.baseline.get(step.decision.key, 0.0) for step in trajectory}
            for step, target in zip(trajectory, returns, strict=True):
                decision = step.decision
                preferences = self.preferences.setdefault(decision.key, [0.0] * len(game.Action))
                advantage = target - old_baselines[decision.key]
                for action, probability in zip(decision.actions, decision.probabilities, strict=True):
                    preferences[action] += self.parameter * advantage * ((action == decision.action) - probability)
                count = self.baseline_counts.get(decision.key, 0) + 1
                baseline = self.baseline.get(decision.key, 0.0)
                self.baseline[decision.key] = baseline + (target - baseline) / count
                self.baseline_counts[decision.key] = count
            return
        for step, target in zip(trajectory, returns, strict=True):
            decision = step.decision
            initial = self.parameter if self.method == "optimistic" else 0.0
            values = self.q.setdefault(decision.key, [initial] * len(game.Action))
            counts = self.updates.setdefault(decision.key, [0] * len(game.Action))
            counts[decision.action] += 1
            rate = 0.1 if self.method == "optimistic" else 1.0 / counts[decision.action]
            values[decision.action] += rate * (target - values[decision.action])


def episode_seed(seed: int, run: int, episode: int) -> int:
    """Stable seed independent of policy, parameter, worker count, or scheduling."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{run}:{episode}".encode()).digest()[:8])


def run_episode(environment: game.HexEnvironment, agent: EpisodicAgent, *, seed: int) -> game.EpisodeResult:
    if environment.board != agent.board or environment.rules != agent.rules:
        raise ValueError("Learner and environment must share the map and rules")
    state = environment.reset(seed=seed)
    trajectory: list[Transition] = []
    total = 0.0
    while True:
        decision = agent.decide(state)
        result = environment.step(decision.action)
        trajectory.append(Transition(decision, result.reward))
        total += result.reward
        state = result.state
        if result.done:
            agent.update_episode(trajectory)
            return game.EpisodeResult(state.outcome, state.turns_elapsed, len(trajectory), total)


@dataclass
class Block:
    end_episode: int = 0
    episodes: int = 0
    reward_sum: float = 0.0
    successes: int = 0
    deaths: int = 0
    timeouts: int = 0
    turns: int = 0
    successful_turns: int = 0
    decisions: int = 0

    def add(self, episode: game.EpisodeResult) -> None:
        if episode.outcome is game.Outcome.RUNNING:
            raise ValueError("Only completed episodes contribute to study scores")
        self.episodes += 1
        self.reward_sum += episode.reward
        self.turns += episode.turns
        self.decisions += episode.decisions
        self.successes += episode.outcome is game.Outcome.SUCCESS
        self.deaths += episode.outcome is game.Outcome.DEATH
        self.timeouts += episode.outcome is game.Outcome.TIMEOUT
        if episode.outcome is game.Outcome.SUCCESS:
            self.successful_turns += episode.turns


@dataclass(frozen=True)
class TrialTask:
    method: Method
    parameter: float
    run: int
    episodes: int
    seed: int
    map_seed: int
    max_turns: int
    block_size: int
    signature: str

    @property
    def filename(self) -> str:
        return f"{self.method}_{self.parameter!r}_run{self.run}.json"


@dataclass(frozen=True)
class TrialResult:
    task: TrialTask
    blocks: tuple[Block, ...]
    seconds: float

    @property
    def average_reward(self) -> float:
        return sum(block.reward_sum for block in self.blocks) / self.task.episodes


@dataclass(frozen=True)
class SweepPoint:
    method: Method
    parameter: float
    average_reward: float
    standard_error: float | None
    success_rate: float
    death_rate: float
    timeout_rate: float
    mean_success_turns: float | None
    mean_turns: float


@dataclass(frozen=True)
class StudyResult:
    episodes: int
    runs: int
    parameters: tuple[float, ...]
    seed: int
    map_seed: int
    max_turns: int
    block_size: int
    signature: str
    trials: tuple[TrialResult, ...]

    @property
    def configurations(self) -> tuple[tuple[Method, float], ...]:
        return configurations(self.parameters)

    @property
    def complete(self) -> bool:
        return len(self.trials) == len(self.configurations) * self.runs

    def points(self) -> tuple[SweepPoint, ...]:
        points: list[SweepPoint] = []
        for method, parameter in self.configurations:
            trials = [
                trial for trial in self.trials if (trial.task.method, trial.task.parameter) == (method, parameter)
            ]
            if len(trials) != self.runs:
                continue  # Never pretend a partly completed multi-run point is complete.
            scores = [trial.average_reward for trial in trials]
            average = mean(scores)
            se = (
                math.sqrt(sum((score - average) ** 2 for score in scores) / (self.runs * (self.runs - 1)))
                if self.runs > 1
                else None
            )
            blocks = [block for trial in trials for block in trial.blocks]
            count = self.episodes * self.runs
            successes = sum(block.successes for block in blocks)
            points.append(
                SweepPoint(
                    method,
                    parameter,
                    average,
                    se,
                    successes / count,
                    sum(block.deaths for block in blocks) / count,
                    sum(block.timeouts for block in blocks) / count,
                    sum(block.successful_turns for block in blocks) / successes if successes else None,
                    sum(block.turns for block in blocks) / count,
                )
            )
        return tuple(points)


def configurations(parameters: tuple[float, ...]) -> tuple[tuple[Method, float], ...]:
    if not parameters or any(not math.isfinite(p) or not 1 / 128 <= p <= 4 for p in parameters):
        raise ValueError("Parameters must be finite and between 1/128 and 4")
    if tuple(sorted(set(parameters))) != parameters:
        raise ValueError("Parameters must be unique and increasing")
    # Interleave methods so partial progress does not consist of only one family.
    return tuple(
        (method, parameter)
        for parameter in parameters
        for method in METHODS
        if method not in ("epsilon_greedy", "gradient") or parameter <= 1
    )


def atomic_text(path: Path, text: str) -> None:
    """Never truncate a completed result in place. Each path has a single writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_trial(result: TrialResult) -> None:
    end = 0
    for block in result.blocks:
        integers = (
            block.end_episode,
            block.episodes,
            block.successes,
            block.deaths,
            block.timeouts,
            block.turns,
            block.successful_turns,
            block.decisions,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("Invalid checkpoint counts")
        if block.episodes != min(result.task.block_size, result.task.episodes - end) or block.episodes < 1:
            raise ValueError("Invalid checkpoint block length")
        end += block.episodes
        if block.end_episode != end or block.successes + block.deaths + block.timeouts != block.episodes:
            raise ValueError("Incomplete checkpoint episode counts")
        if not block.episodes <= block.turns <= block.episodes * result.task.max_turns:
            raise ValueError("Invalid checkpoint turn counts")
        if (
            not block.turns <= block.decisions <= 2 * block.turns
            or not block.successes <= block.successful_turns <= block.turns
        ):
            raise ValueError("Invalid checkpoint decision/success counts")
        expected_reward = 100 * block.successes - 1000 * (block.deaths + block.timeouts) - block.turns
        if not math.isfinite(block.reward_sum) or block.reward_sum != expected_reward:
            raise ValueError("Checkpoint return disagrees with game outcomes")
    if end != result.task.episodes or not math.isfinite(result.seconds) or result.seconds < 0:
        raise ValueError("Only complete configuration/run checkpoints can be resumed")


def load_trial(path: Path, task: TrialTask) -> TrialResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["task"] != asdict(task):
            raise ValueError("Checkpoint settings or source signature differ")
        result = TrialResult(task, tuple(Block(**block) for block in payload["blocks"]), float(payload["seconds"]))
        _validate_trial(result)
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint {path}: {error}") from error


def run_trial(task: TrialTask, output_dir: Path | None = None) -> TrialResult:
    """One independent fresh learner; keep at most one episode trajectory in memory."""
    board, rules = game.HexMap.generate(task.map_seed), game.Rules(max_turns=task.max_turns)
    environment = game.HexEnvironment(board, rules=rules)
    # Common initial policy RNG seeds as well; methods naturally consume draws differently.
    agent = EpisodicAgent(board, rules, task.method, task.parameter, seed=episode_seed(task.seed, task.run, -1))
    blocks: list[Block] = []
    block = Block()
    started = perf_counter()
    for episode in range(1, task.episodes + 1):
        block.add(run_episode(environment, agent, seed=episode_seed(task.seed, task.run, episode)))
        if episode % task.block_size == 0 or episode == task.episodes:
            block.end_episode = episode
            blocks.append(block)
            block = Block()
        if episode % 10000 == 0:
            print(
                f"[{task.method} {task.parameter:g}, run {task.run + 1}] {episode:,}/{task.episodes:,} episodes; {perf_counter() - started:.1f}s",
                flush=True,
            )
    result = TrialResult(task, tuple(blocks), perf_counter() - started)
    _validate_trial(result)
    if output_dir is not None:
        atomic_text(output_dir / "checkpoints" / task.filename, json.dumps(asdict(result), allow_nan=False))
    return result


def plot_results(result: StudyResult, directory: Path) -> None:
    """Project-8-style parameter curves; uncertainty only across independent runs."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points = result.points()
    if not points:
        return
    for learning_curve in (False, True):
        figure = Figure(figsize=(11, 7))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for method in METHODS:
            selected = sorted((point for point in points if point.method == method), key=lambda point: point.parameter)
            if not selected:
                continue
            if learning_curve:
                best = max(selected, key=lambda point: point.average_reward)
                trials = [
                    trial
                    for trial in result.trials
                    if trial.task.method == method and trial.task.parameter == best.parameter
                ]
                x = [float(block.end_episode) for block in trials[0].blocks]
                cumulative = [[0.0] * len(x) for _ in trials]
                for run, trial in enumerate(trials):
                    total = 0.0
                    for index, block in enumerate(trial.blocks):
                        total += block.reward_sum
                        cumulative[run][index] = total / block.end_episode
                y = [mean(values[index] for values in cumulative) for index in range(len(x))]
                axis.plot(x, y, color=COLORS[method], label=f"{LABELS[method]} = {best.parameter:g}")
            else:
                x = [point.parameter for point in selected]
                y = [point.average_reward for point in selected]
                axis.plot(x, y, marker="o", color=COLORS[method], label=LABELS[method])
                if result.runs > 1:
                    errors = [point.standard_error or 0.0 for point in selected]
                    axis.fill_between(
                        x,
                        [value - error for value, error in zip(y, errors, strict=True)],
                        [value + error for value, error in zip(y, errors, strict=True)],
                        color=COLORS[method],
                        alpha=0.15,
                    )
        status = "complete" if result.complete else f"PARTIAL: {len(points)}/{len(result.configurations)} settings"
        axis.set_title(
            f"Hex-world episodic policy study — {status}\n{result.runs} independent run(s), {result.episodes:,} episodes each"
        )
        if learning_curve:
            axis.set_xlabel("Training episodes (best observed parameter per completed method)")
            axis.set_ylabel("Cumulative average episode return since episode 1")
            name = "learning_curves.png"
        else:
            axis.set_xscale("log", base=2)
            axis.set_xticks(
                result.parameters,
                [f"1/{round(1 / p)}" if p < 1 and (1 / p).is_integer() else f"{p:g}" for p in result.parameters],
            )
            axis.set_xlabel(r"Parameter ($\varepsilon$, $\alpha$, $Q_0$, or $c$); learning rewards scaled by 1/1000")
            axis.set_ylabel(f"Average total reward over the first {result.episodes:,} episodes (original units)")
            name = "parameter_study.png"
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)
        figure.tight_layout()
        temporary = directory / f".{name}.tmp"
        try:
            figure.savefig(temporary, format="png", dpi=160)
            temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)


def save_report(result: StudyResult, directory: Path) -> None:
    """Regenerate small aggregate artifacts; incomplete studies are explicitly labeled."""
    directory.mkdir(parents=True, exist_ok=True)
    points = result.points()
    metadata = {key: value for key, value in asdict(result).items() if key != "trials"}
    metadata.update(
        complete=result.complete, completed_runs=len(result.trials), configurations=len(result.configurations)
    )
    atomic_text(
        directory / "results.json",
        json.dumps({"settings": metadata, "points": [asdict(point) for point in points]}, indent=2, allow_nan=False),
    )
    buffer = StringIO()
    writer = csv.writer(buffer)
    fields = tuple(SweepPoint.__dataclass_fields__)
    writer.writerow((*fields, "episodes", "runs", "seed", "map_seed"))
    for point in points:
        writer.writerow((*asdict(point).values(), result.episodes, result.runs, result.seed, result.map_seed))
    atomic_text(directory / "parameter_scores.csv", buffer.getvalue())
    plot_results(result, directory)
    status = "Complete" if result.complete else "PARTIAL — do not interpret missing settings as zero scores"
    lines = [
        "# Project 11 — Episodic Policy Parameter Study",
        "",
        f"**Status: {status}.**",
        "",
        f"Completed configuration/runs: **{len(result.trials)} / {len(result.configurations) * result.runs}**.",
        f"Each complete plotted setting averages all of the first **{result.episodes:,} episodes per run**, across **{result.runs} independent run(s)**.",
        f"Master seed: `{result.seed}`; fixed map seed: `{result.map_seed}`; horizon: `{result.max_turns}` turns; learning-curve block: `{result.block_size}` episodes.",
        f"Source/settings signature: `{result.signature}`.",
        "",
        "## Concept: actions are not episodes",
        "",
        "The agent makes multiple movement, rest, and combat decisions before success, death, or timeout. The score is the **sum of all environment rewards in a complete episode**, not a reward per move and not the final reward alone. Faster failures do not become better merely because they contain fewer decisions.",
        "Training includes exploration and all failures. A successful episode lasting T turns returns 100 − T; death or timeout returns −1000 − T. For N episodes and K independent runs, the plotted score is the mean of each run's sum of N episode returns divided by N. This is not a frozen-policy evaluation or a final-window score.",
        "",
        "## Methods and experimental controls",
        "",
        "| Method | Parameter | Episode-end update |",
        "| --- | --- | --- |",
        "| Epsilon-greedy | epsilon, 1/128 through 1 | Every-visit Monte Carlo sample-average action values |",
        "| Gradient bandit adapted to episodes | alpha, 1/128 through 1 | REINFORCE, frozen within-episode softmax, pre-episode state baseline |",
        "| Optimistic greedy | initial Q, 1/128 through 4 | Every-visit Monte Carlo, fixed step size 0.1; no epsilon exploration |",
        "| State-local UCB | c, 1/128 through 4 | Every-visit Monte Carlo sample averages; per-state action-selection counts |",
        "",
        "The full grid is powers of two; a custom subset may be selected. Legal actions are masked for every method. Greedy/UCB ties and untried UCB actions are selected uniformly. UCB uses Q(s,a) + c sqrt(log(N(s)+1)/N(s,a)), not a global episode counter. Its bonus is a heuristic here, not a stationary-bandit confidence guarantee.",
        "All methods use the same state aggregation as the demo: exact player position, HP, movement allowance and enemy HP; enemy position only within three hexes; horizons of ten or more turns share a bucket. Aliasing remains: this is a restricted policy comparison, not a solution of the full observable game.",
        "Learners start fresh for every parameter/run; no values are shared. Each episode is reseeded by master seed, run, and episode number, independent of the method and worker schedule. Thus initial enemy placement is paired. Enemy trajectories can still differ when policies consume random draws differently or end at different times.",
        "",
        "### Credit assignment and scale",
        "",
        "At the end of an episode, every decision receives its **return-to-go**, not just its immediate reward and not the entire starting return regardless of time. There is no future-value bootstrap. Unlike the quick demo, this study uses **no distance-potential shaping**.",
        "Learning rewards are divided by **1000 for all methods**. This keeps the Figure-2.6-style numerical grid meaningful next to the game's −1000 failure penalty. Plot scores remain in original game units. Alpha, c, and initial Q therefore refer to this fixed learning scale; changing it changes the meaning of the grid. The smallest positive initial values are not necessarily optimistic relative to a successful policy's value.",
        "The episodic gradient update sums alpha (G_t − b_old(s_t)) [one_hot(A_t) − pi_old(.|s_t)] over the episode. Probabilities are those actually used to sample actions. Baselines exclude the entire current episode, even if a state repeats, and are updated afterward conceptually. The gradient is summed, not divided by episode length. This is **REINFORCE**, not the ordinary immediate-reward gradient-bandit update.",
        "This comparison changes learning rules as well as selection behavior, as Project 8 did. It cannot isolate an effect caused only by the action selector.",
        "",
    ]
    if points:
        lines.extend(
            [
                "## Parameter sensitivity",
                "",
                "![Average full-episode return over the training horizon versus parameter](parameter_study.png)",
                "",
                "**Results.** The horizontal axis is the parameter on a base-two logarithmic scale. The vertical axis averages the complete training horizon, including early learning and persistent exploration. Red is epsilon-greedy, green is episodic policy gradient, orange is optimistic greedy, and blue is state-local UCB. Higher is better. No optimal-policy reference line is claimed.",
                (
                    "Bands show ±1 standard error across independent run means."
                    if result.runs > 1
                    else "This is a **single-run-per-setting study**: no uncertainty bands are drawn. Episode samples within one learning run are not treated as independent replications."
                ),
                "",
                "### Best completed sampled settings",
                "",
                "| Method | Parameter | Mean episode return | Success | Death | Timeout | Mean turns on success |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            selected = [point for point in points if point.method == method]
            if not selected:
                continue
            best = max(selected, key=lambda point: point.average_reward)
            turns = f"{best.mean_success_turns:.2f}" if best.mean_success_turns is not None else "n/a"
            lines.append(
                f"| {NAMES[method]} | {best.parameter:g} | {best.average_reward:.4f} | {best.success_rate:.2%} | {best.death_rate:.2%} | {best.timeout_rate:.2%} | {turns} |"
            )
        leader = max(points, key=lambda point: point.average_reward)
        lines.extend(
            [
                "",
                f"Among completed settings, **{NAMES[leader.method]} at {leader.parameter:g}** has the highest observed mean ({leader.average_reward:.4f}).",
            ]
        )
        for method in METHODS:
            selected = [point for point in points if point.method == method]
            if len(selected) > 1:
                worst = min(selected, key=lambda point: point.average_reward)
                best = max(selected, key=lambda point: point.average_reward)
                lines.append(
                    f"- {NAMES[method]} spans {worst.average_reward:.4f} (parameter {worst.parameter:g}) to {best.average_reward:.4f} ({best.parameter:g}) across completed settings."
                )
        lines.extend(
            [
                "",
                "These are observed maxima among completed sampled settings, not universal optima. Selection uses the same training data being displayed, not a held-out test set. A partial study's leaders can change as remaining settings finish.",
                "",
                "## Learning across episodes",
                "",
                "![Cumulative average return for each method's best completed setting](learning_curves.png)",
                "",
                "**Results.** Each curve includes every episode since episode 1; its final value equals its setting's point in the parameter chart. These are cumulative means, not raw per-episode rewards or smoothed final-policy evaluations. One setting per method is selected retrospectively by the full-horizon score. Curves can conceal recent regressions, so they should not be read as instantaneous performance.",
                "",
                "## Interpretation and limits",
                "",
                "Epsilon spends a continuing fraction of decisions exploring, potentially trading shorter learned routes for additional risk. UCB exploration depends on visits to each aggregate state; its estimates change as later behavior changes, violating ordinary stationary-bandit assumptions. Optimistic initialization encourages trying alternatives only until the prior estimates are corrected. REINFORCE directly changes action probabilities, with alpha controlling the response to noisy episodic returns and a baseline reducing, not removing, that noise.",
                "These mechanisms are hypotheses for explaining measured differences, not proof that a particular bend in one seed's curve has a unique cause. Compare success/death/timeout rates alongside travel time: a fast policy with frequent deaths can have a very poor mean return. Repeat with more independent runs and additional fixed maps before generalizing.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No configuration has completed all requested runs yet. Charts will be generated from completed results only.",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction and files",
            "",
            f"```bash\npython -m reinforcement_learning.projects.project011 --study --episodes {result.episodes} --runs {result.runs} --seed {result.seed} --map-seed {result.map_seed} --max-turns {result.max_turns} --block-size {result.block_size} --parameters {' '.join(str(p) for p in result.parameters)} --output-dir \"{directory.parent.as_posix()}\"\n```",
            "",
            "Add `--workers 4` to parallelize independent configurations/runs. The same command works in PowerShell. It requires Matplotlib for PNG plots. Use the same `--output-dir` on resume; study files live in its `study/` subdirectory.",
            "`parameter_scores.csv` and `results.json` contain aggregate scores. Completed-run checkpoints contain only block summaries, never policies, raw trajectories, or large per-episode tables. Writes replace temporary files atomically. Matching completed runs are reused; an interrupted unfinished run restarts from episode 1. Changed source/settings require a different output directory. Only one study process should write a given directory.",
            "",
        ]
    )
    atomic_text(directory / "report.md", "\n".join(lines))


def run_study(
    *,
    output_dir: Path,
    episodes: int = 100000,
    runs: int = 1,
    seed: int = 42,
    map_seed: int = 42,
    max_turns: int = 500,
    workers: int = 1,
    parameters: tuple[float, ...] = PARAMETERS,
    block_size: int = 1000,
) -> StudyResult:
    parameters = tuple(float(parameter) for parameter in parameters)
    grid = configurations(parameters)
    if any(type(value) is not int or value < 1 for value in (episodes, runs, max_turns, workers, block_size)):
        raise ValueError("Episodes, runs, turn limit, workers, and block size must be positive integers")
    # Check plotting availability before launching a potentially long experiment.
    from matplotlib.figure import Figure

    del Figure
    sources = hashlib.sha256(Path(game.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()
    settings = {
        "version": 1,
        "episodes": episodes,
        "runs": runs,
        "seed": seed,
        "map_seed": map_seed,
        "max_turns": max_turns,
        "parameters": list(parameters),
        "block_size": block_size,
        "reward_scale": REWARD_SCALE,
        "source_hash": sources,
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Study settings/source changed; use a different --output-dir rather than mixing results")
    else:
        atomic_text(manifest, json.dumps(settings, indent=2))
    tasks = [
        TrialTask(method, parameter, run, episodes, seed, map_seed, max_turns, block_size, signature)
        for method, parameter in grid
        for run in range(runs)
    ]
    completed: list[TrialResult] = []
    pending: list[TrialTask] = []
    for task in tasks:
        checkpoint = output_dir / "checkpoints" / task.filename
        if checkpoint.exists():
            completed.append(load_trial(checkpoint, task))
        else:
            pending.append(task)

    def snapshot() -> StudyResult:
        ordered = tuple(sorted(completed, key=lambda trial: (trial.task.parameter, trial.task.method, trial.task.run)))
        return StudyResult(episodes, runs, parameters, seed, map_seed, max_turns, block_size, signature, ordered)

    def record(trial: TrialResult) -> None:
        completed.append(trial)
        print(
            f"Completed {trial.task.method} {trial.task.parameter:g}, run {trial.task.run + 1}: {trial.average_reward:.4f} mean episode return ({trial.seconds:.1f}s)",
            flush=True,
        )
        save_report(snapshot(), output_dir)

    save_report(snapshot(), output_dir)
    print(
        f"Study: {len(grid)} settings × {runs} run(s) × {episodes:,} episodes; reusing {len(completed)} completed run(s).",
        flush=True,
    )
    if workers == 1:
        for task in pending:
            record(run_trial(task, output_dir))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            try:
                futures = [pool.submit(run_trial, task, output_dir) for task in pending]
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                # Stop promptly on interruption/failure instead of waiting for
                # every queued long trial. Completed atomic files remain usable.
                pool.terminate_workers()
                raise
    result = snapshot()
    print(f"Saved study report: {output_dir / 'report.md'}", flush=True)
    return result
