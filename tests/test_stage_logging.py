from concurrent.futures import ThreadPoolExecutor

from surrogate_rollout.optimization.stage_logging import IterationStageLogger


def test_iteration_stage_log_lines_are_thread_safe_and_mirrored(tmp_path, capsys):
    logger = IterationStageLogger(str(tmp_path / "iteration.log"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(
            lambda index: logger.emit(
                event="END", stage="fixture",
                message=f"video-{index} 끝", result=index),
            range(40)))

    lines = (tmp_path / "iteration.log").read_text().splitlines()
    stdout_lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(stdout_lines) == 40
    assert set(lines) == set(stdout_lines)
    assert all("[END] [fixture]" in line and '"result":' in line
               for line in lines)
