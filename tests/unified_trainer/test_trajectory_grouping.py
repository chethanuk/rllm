import pytest

from rllm.trainer.algorithms.transform import _build_trajectory_groups, _get_transform_metrics
from rllm.types import Episode, Step, Trajectory, TrajectoryGroup


def _episodes(row_id_template, n_rows=16, rollout_n=2):
    eps = []
    for row in range(n_rows):
        row_id = row_id_template.format(row=row)
        for r in range(rollout_n):
            traj = Trajectory(name="agent_0", steps=[Step(reward=1.0)])
            traj.reward = 1.0
            eps.append(Episode(id=f"{row_id}:{r}", trajectories=[traj]))
    return eps


@pytest.mark.parametrize(
    "row_id_template",
    [
        "task{row}",  # no colon (must not regress)
        "aeread:integration-v1:case{row}:s1200",  # colons in the row id
        "a:b:c:d:e:{row}",  # many colons
    ],
)
def test_one_group_per_row_regardless_of_colons_in_row_id(row_id_template):
    eps = _episodes(row_id_template)
    metrics = _get_transform_metrics(eps, _build_trajectory_groups(eps))
    assert metrics["groups/num_groups"] == 16
    assert metrics["groups/avg_group_size"] == 2.0
    assert len({e.task_id for e in eps}) == 16


@pytest.mark.parametrize(
    ("episode_id", "task_id", "rollout_idx"),
    [
        ("task0:0", "task0", "0"),
        ("a:b:c:0", "a:b:c", "0"),
        ("aeread:integration-v1:case01:s1200:7", "aeread:integration-v1:case01:s1200", "7"),
    ],
)
def test_episode_id_splits_on_the_last_colon(episode_id, task_id, rollout_idx):
    ep = Episode(id=episode_id)
    assert ep.task_id == task_id
    assert ep.rollout_idx == rollout_idx


def test_group_role_is_the_trajectory_name_not_an_id_fragment():
    group = TrajectoryGroup(trajectories=[], group_id="aeread:integration-v1:case01:s1200:agent_0")
    assert group.group_role == "agent_0"
    assert group.task_id == "aeread:integration-v1:case01:s1200"


def test_id_without_a_colon_keeps_current_behaviour():
    ep = Episode(id="bare-uuid-no-colon")
    assert ep.task_id == "bare-uuid-no-colon"
    with pytest.raises(IndexError):
        _ = ep.rollout_idx
