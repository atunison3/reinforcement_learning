"""Episodic policy-credit, fair-score, checkpoint, plot, and runner tests."""

import json
import math
import unittest
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from reinforcement_learning.projects import project011 as game
from reinforcement_learning.projects.project011_study import (
    METHODS,
    PARAMETERS,
    Block,
    Decision,
    EpisodicAgent,
    StudyResult,
    Transition,
    TrialResult,
    TrialTask,
    atomic_text,
    configurations,
    episode_seed,
    load_trial,
    run_episode,
    run_study,
    run_trial,
    save_report,
)


def board() -> game.HexMap:
    return game.HexMap(((game.Terrain.PLAINS,) * 5,), game.Hex(0, 0), game.Hex(2, 0))


def task(*, run: int = 0, episodes: int = 2, block_size: int = 1) -> TrialTask:
    return TrialTask("epsilon_greedy", 0.125, run, episodes, 42, 42, 500, block_size, "test-signature")


def complete_trial(*, run: int = 0, success: bool = True) -> TrialResult:
    block = Block()
    outcome = game.Outcome.SUCCESS if success else game.Outcome.DEATH
    block.add(game.EpisodeResult(outcome, 5, 8, 95 if success else -1005))
    block.end_episode = 1
    return TrialResult(task(run=run, episodes=1), (block,), 0.5)


class TestEpisodicPolicies(unittest.TestCase):
    def setUp(self) -> None:
        self.board = board()
        self.rules = game.Rules()
        self.state = game.GameState(self.board.city1, 100, None, 0)
        self.key = game.learning_key(self.board, self.rules, self.state)
        self.actions = game.allowed_actions(self.board, self.state)

    def test_default_grid_matches_project_eight_with_thirty_six_settings(self) -> None:
        grid = configurations(PARAMETERS)
        self.assertEqual(len(grid), 36)
        for method, expected in (("epsilon_greedy", 8), ("gradient", 8), ("optimistic", 10), ("ucb", 10)):
            self.assertEqual(sum(name == method for name, _ in grid), expected)
        for parameters in ((), (0.0,), (8.0,), (float("nan"),), (1.0, 0.5), (0.5, 0.5)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                configurations(parameters)

    def test_every_method_masks_illegal_actions_including_high_illegal_values(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                agent = EpisodicAgent(self.board, self.rules, method, 0.125, seed=3)
                agent.q[self.key] = [0.0] * 8
                agent.preferences[self.key] = [0.0] * 8
                agent.q[self.key][game.Action.NORTHWEST] = 1000000
                agent.preferences[self.key][game.Action.NORTHWEST] = 1000000
                for _ in range(100):
                    self.assertIn(agent.decide(self.state).action, self.actions)
                if method == "ucb":
                    self.assertEqual(agent.counts[self.key][game.Action.NORTHWEST], 0)

    def test_epsilon_one_explores_all_legal_actions(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "epsilon_greedy", 1, seed=5)
        observed = {agent.decide(self.state).action for _ in range(100)}
        self.assertEqual(observed, set(self.actions))

    def test_ucb_tries_unseen_actions_before_values_then_uses_local_counts(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "ucb", 1, seed=1)
        agent.q[self.key] = [0.0] * 8
        agent.q[self.key][game.Action.EAST] = 100
        agent.counts[self.key] = [0] * 8
        agent.counts[self.key][game.Action.EAST] = 10
        for _ in range(2):
            self.assertNotEqual(agent.decide(self.state).action, game.Action.EAST)
        self.assertEqual(agent.counts[self.key][game.Action.WEST], 1)
        self.assertEqual(agent.counts[self.key][game.Action.REST], 1)
        agent.q[self.key] = [0.0] * 8
        agent.counts[self.key][game.Action.EAST] = 1
        agent.counts[self.key][game.Action.WEST] = 4
        agent.counts[self.key][game.Action.REST] = 4
        self.assertEqual(agent.decide(self.state).action, game.Action.EAST)
        self.assertEqual(agent.counts[self.key][game.Action.EAST], 2)
        other = replace(self.state, player=game.Hex(1, 0))
        other_key = game.learning_key(self.board, self.rules, other)
        agent.q[other_key] = [0.0] * 8
        agent.q[other_key][game.Action.EAST] = 10000
        with patch.object(agent.rng, "choice", side_effect=lambda choices: choices[-1]):
            self.assertEqual(agent.decide(other).action, game.Action.REST)

    def test_every_visit_sample_averages_use_returns_to_go_and_correct_update_counts(self) -> None:
        for method in ("epsilon_greedy", "ucb"):
            with self.subTest(method=method):
                agent = EpisodicAgent(self.board, self.rules, method, 0.125, seed=0)
                decision = Decision(self.key, self.actions, game.Action.EAST)
                agent.counts[self.key] = [0] * 8
                agent.counts[self.key][game.Action.EAST] = 2  # Already sampled twice before learning.
                agent.update_episode((Transition(decision, -1), Transition(decision, 99)))
                self.assertAlmostEqual(agent.q[self.key][game.Action.EAST], (0.098 + 0.099) / 2)
                self.assertEqual(agent.updates[self.key][game.Action.EAST], 2)
                self.assertEqual(agent.q[self.key][game.Action.REST], 0)

    def test_optimistic_greedy_uses_tested_initial_value_and_fixed_point_one_update(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "optimistic", 2, seed=0)
        decision = agent.decide(self.state)
        self.assertEqual(agent.q[self.key], [2.0] * 8)
        agent.update_episode((Transition(decision, 100),))
        self.assertAlmostEqual(agent.q[self.key][decision.action], 2 + 0.1 * (0.1 - 2))
        for candidate in self.actions:
            if candidate != decision.action:
                self.assertEqual(agent.q[self.key][candidate], 2)

    def test_gradient_uses_stored_probabilities_and_old_state_baseline(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "gradient", 0.5, seed=0)
        actions = (game.Action.EAST, game.Action.WEST)
        decision = Decision(self.key, actions, game.Action.EAST, (0.75, 0.25))
        agent.preferences[self.key] = [0.0] * 8
        agent.preferences[self.key][game.Action.EAST] = math.log(3)
        agent.preferences[self.key][game.Action.ATTACK] = 1234
        agent.baseline[self.key] = 0.2
        agent.baseline_counts[self.key] = 1
        agent.update_episode((Transition(decision, 1000),))
        self.assertAlmostEqual(agent.preferences[self.key][game.Action.EAST], math.log(3) + 0.1)
        self.assertAlmostEqual(agent.preferences[self.key][game.Action.WEST], -0.1)
        self.assertEqual(agent.preferences[self.key][game.Action.ATTACK], 1234)
        self.assertAlmostEqual(agent.baseline[self.key], 0.6)

    def test_repeated_gradient_states_do_not_leak_current_returns_into_baseline_or_resample_policy(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "gradient", 1, seed=0)
        decision = Decision(self.key, (game.Action.EAST, game.Action.WEST), game.Action.EAST, (0.5, 0.5))
        agent.update_episode((Transition(decision, 0), Transition(decision, 1000)))
        self.assertEqual(agent.preferences[self.key][game.Action.EAST], 1)
        self.assertEqual(agent.preferences[self.key][game.Action.WEST], -1)
        self.assertEqual(agent.baseline[self.key], 1)
        self.assertEqual(agent.baseline_counts[self.key], 2)

    def test_softmax_is_stable_over_large_legal_preferences(self) -> None:
        agent = EpisodicAgent(self.board, self.rules, "gradient", 0.125, seed=0)
        agent.preferences[self.key] = [1000.0] * 8
        probabilities = agent.probabilities(self.key, self.actions)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertEqual(probabilities, (1 / 3, 1 / 3, 1 / 3))
        with self.assertRaises(ValueError):
            agent.probabilities(self.key, ())

    def test_invalid_parameters_and_episodes(self) -> None:
        for method in METHODS:
            for parameter in (0, -1, float("nan"), float("inf")):
                with self.subTest(method=method, parameter=parameter), self.assertRaises(ValueError):
                    EpisodicAgent(self.board, self.rules, method, parameter, seed=0)
        for method in ("epsilon_greedy", "gradient"):
            with self.assertRaises(ValueError):
                EpisodicAgent(self.board, self.rules, method, 2, seed=0)
        agent = EpisodicAgent(self.board, self.rules, "epsilon_greedy", 0.125, seed=0)
        decision = Decision(self.key, self.actions, game.Action.EAST)
        for steps in (
            (),
            (Transition(decision, float("nan")),),
            (Transition(replace(decision, action=game.Action.ATTACK), 1),),
        ):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                agent.update_episode(steps)
        self.assertFalse(agent.q)

    def test_real_episode_learns_once_at_end_and_reports_original_total_reward(self) -> None:
        environment = game.HexEnvironment(self.board)
        agent = EpisodicAgent(self.board, self.rules, "epsilon_greedy", 0.125, seed=0)
        reset = environment.reset

        def east(state: game.GameState) -> Decision:
            self.assertFalse(agent.q)  # No within-episode learning.
            return Decision(
                game.learning_key(self.board, self.rules, state),
                game.allowed_actions(self.board, state),
                game.Action.EAST,
            )

        with (
            patch.object(environment, "reset", side_effect=lambda **kwargs: reset(barbarian_health=0, **kwargs)),
            patch.object(agent, "decide", side_effect=east),
            patch.object(agent, "update_episode", wraps=agent.update_episode) as learn,
        ):
            result = run_episode(environment, agent, seed=123)
        self.assertEqual(result, game.EpisodeResult(game.Outcome.SUCCESS, 1, 2, 99))
        learn.assert_called_once()
        self.assertEqual(len(learn.call_args.args[0]), 2)
        for values in agent.q.values():
            self.assertEqual(values[game.Action.EAST], 0.099)

    def test_pairing_seeds_restarts_environment_independent_of_previous_episode_length(self) -> None:
        left, right = game.HexEnvironment(), game.HexEnvironment(seed=999)
        for episode in range(5):
            seed = episode_seed(42, 0, episode)
            self.assertEqual(left.reset(seed=seed), right.reset(seed=seed))
            left.step(game.Action.REST)
        self.assertNotEqual(episode_seed(42, 0, 0), episode_seed(42, 1, 0))


class TestStudyReports(unittest.TestCase):
    def test_episode_means_are_not_decision_weighted_or_success_only(self) -> None:
        block = Block()
        block.add(game.EpisodeResult(game.Outcome.SUCCESS, 5, 10, 95))
        block.add(game.EpisodeResult(game.Outcome.DEATH, 2, 3, -1002))
        block.add(game.EpisodeResult(game.Outcome.TIMEOUT, 500, 500, -1500))
        block.end_episode = 3
        trial = TrialResult(task(episodes=3, block_size=3), (block,), 1)
        self.assertAlmostEqual(trial.average_reward, -2407 / 3)
        result = StudyResult(3, 1, (0.125,), 42, 42, 500, 3, "signature", (trial,))
        point = result.points()[0]
        self.assertAlmostEqual(point.success_rate, 1 / 3)
        self.assertAlmostEqual(point.death_rate, 1 / 3)
        self.assertAlmostEqual(point.timeout_rate, 1 / 3)
        self.assertEqual(point.mean_success_turns, 5)
        self.assertIsNone(point.standard_error)
        with self.assertRaises(ValueError):
            block.add(game.EpisodeResult(game.Outcome.RUNNING, 0, 0, 0))

    def test_uncertainty_uses_independent_run_means_not_correlated_episodes(self) -> None:
        trials = (complete_trial(), complete_trial(run=1, success=False))
        result = StudyResult(1, 2, (0.125,), 42, 42, 500, 1, "signature", trials)
        point = result.points()[0]
        self.assertEqual(point.average_reward, -455)
        self.assertEqual(point.standard_error, 550)
        self.assertEqual(point.success_rate, 0.5)
        self.assertEqual(replace(result, trials=trials[:1]).points(), ())
        self.assertFalse(result.complete)

    def test_trial_seeds_and_block_summaries_are_reproducible(self) -> None:
        short = replace(task(episodes=5, block_size=2), max_turns=3)
        a, b = run_trial(short), run_trial(short)
        self.assertEqual(replace(a, seconds=0), replace(b, seconds=0))
        self.assertEqual([block.episodes for block in a.blocks], [2, 2, 1])
        self.assertEqual([block.end_episode for block in a.blocks], [2, 4, 5])

    def test_atomic_write_preserves_previous_file_when_replacement_fails(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_text(path, "original")
            with patch.object(Path, "replace", side_effect=OSError("interrupted write")), self.assertRaises(OSError):
                atomic_text(path, "replacement")
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_checkpoint_rejects_partial_counts_bad_rewards_and_different_settings(self) -> None:
        original = complete_trial()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trial.json"
            payload = asdict(original)
            atomic_text(path, json.dumps(payload))
            self.assertEqual(load_trial(path, original.task), original)
            with self.assertRaises(ValueError):
                load_trial(path, replace(original.task, episodes=2))
            for updates in (
                {"episodes": 0},
                {"reward_sum": 1000},
                {"successes": 2},
                {"end_episode": 2},
                {"decisions": 1000},
            ):
                with self.subTest(updates=updates):
                    changed = asdict(original)
                    changed["blocks"][0].update(updates)
                    atomic_text(path, json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_trial(path, original.task)
            atomic_text(path, json.dumps({**payload, "blocks": []}))
            with self.assertRaises(ValueError):
                load_trial(path, original.task)

    def test_reports_plot_actual_horizon_and_label_single_run_and_partial_results(self) -> None:
        result = StudyResult(1, 1, (0.125,), 42, 42, 500, 1, "signature", (complete_trial(),))
        with TemporaryDirectory() as directory:
            save_report(result, Path(directory))
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("PARTIAL", report)
            self.assertIn("first **1 episodes per run**", report)
            self.assertIn("single-run-per-setting", report)
            self.assertIn("95.0000", report)
            self.assertNotIn("first **100,000 episodes per run**", report)
            for name in ("parameter_study.png", "learning_curves.png"):
                path = Path(directory) / name
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertLess(path.stat().st_size, 2_000_000)
            data = json.loads((Path(directory) / "results.json").read_text())
            self.assertFalse(data["settings"]["complete"])
            self.assertIsNone(data["points"][0]["standard_error"])

    def test_complete_runs_resume_without_training_and_setting_changes_are_rejected(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch("sys.stdout", new_callable=StringIO),
            patch("reinforcement_learning.projects.project011_study.plot_results"),
        ):
            output = Path(directory)
            result = run_study(output_dir=output, episodes=2, parameters=(0.125,), max_turns=2, block_size=1)
            self.assertTrue(result.complete)
            self.assertEqual(len(result.points()), 4)
            with patch(
                "reinforcement_learning.projects.project011_study.run_trial",
                side_effect=AssertionError("Should resume"),
            ):
                restored = run_study(output_dir=output, episodes=2, parameters=(0.125,), max_turns=2, block_size=1)
            self.assertEqual(restored, result)
            original = (output / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                run_study(output_dir=output, episodes=3, parameters=(0.125,), max_turns=2, block_size=1)
            self.assertEqual((output / "manifest.json").read_bytes(), original)

    def test_parallel_scheduling_does_not_change_scores(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch("sys.stdout", new_callable=StringIO),
            patch("reinforcement_learning.projects.project011_study.plot_results"),
        ):
            root = Path(directory)
            left = run_study(output_dir=root / "serial", episodes=2, parameters=(0.125,), max_turns=2, block_size=1)
            right = run_study(
                output_dir=root / "parallel", episodes=2, parameters=(0.125,), max_turns=2, block_size=1, workers=2
            )
            self.assertEqual(left.points(), right.points())
            self.assertEqual(left.signature, right.signature)

    def test_project_eleven_cli_routes_to_study_with_one_hundred_thousand_default(self) -> None:
        with (
            patch("sys.argv", ["project011", "--study", "--workers", "2"]),
            patch("reinforcement_learning.projects.project011_study.run_study") as study,
            patch.object(game, "run_episode", side_effect=AssertionError("Not the demo")),
        ):
            game.main()
        study.assert_called_once()
        self.assertEqual(study.call_args.kwargs["episodes"], 100000)
        self.assertEqual(study.call_args.kwargs["workers"], 2)
        self.assertEqual(study.call_args.kwargs["parameters"], PARAMETERS)
        self.assertEqual(study.call_args.kwargs["output_dir"], Path("docs/projects/assets/project_11/study"))


if __name__ == "__main__":
    unittest.main()
