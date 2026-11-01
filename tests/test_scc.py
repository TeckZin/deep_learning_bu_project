"""Exercise SCC submission plumbing with a fake scheduler, without a GPU."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class SCCTests(unittest.TestCase):
    def fixture(self, root):
        executable = root / "env/bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json,os,subprocess,sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:2] == ['-c']:\n"
            "    sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
            "with open(os.environ['COMMAND_CAPTURE'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if 'train' in sys.argv and os.environ.get('FAIL_TRAIN'): sys.exit(23)\n"
        )
        executable.chmod(0o755)
        (root / "env/bin/activate").write_text(f"export PATH={shlex.quote(str(executable.parent))}:\"$PATH\"\n")
        binary = root / "bin/qsub"
        binary.parent.mkdir()
        binary.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
                          "Path(os.environ['QSUB_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
        binary.chmod(0o755)
        run = root / "run"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "crop_size": 128, "batch_size": 2, "train_per_domain": 2,
            "val_per_domain": 1, "seed": 523,
            "training": {"baseline_updates": 10, "refiner_updates": 10}}))
        images = root / "data/Val/Urban/images_png"
        images.mkdir(parents=True)
        (images / "000.png").touch()
        settings = {
            "SCC_PROJECT": "course-project", "SCC_REPO": str(REPO), "SCC_ENV": str(root / "env"),
            "OUTPUT_ROOT": str(root / "output"), "DATA_ROOT": str(root / "data"),
            "ENCODER_WEIGHTS": str(root / "encoder.safetensors"), "SCC_RUN_DIR": str(run),
            "SCC_CPUS": "4", "SCC_GPU_MEMORY": "24G", "SCC_GPU_CAPABILITY": "7.0",
            "SCC_WALLTIME": "04:00:00", "SCC_BASELINE_UPDATES": "200", "SCC_REFINER_UPDATES": "200",
        }
        config = root / "config.sh"
        config.write_text("\n".join(f"export {k}={shlex.quote(v)}" for k, v in settings.items()) + "\n")
        environment = {**os.environ, "SCC_CONFIG": str(config), "COMMAND_CAPTURE": str(root / "commands"),
                       "QSUB_CAPTURE": str(root / "qsub.json"), "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}
        return environment

    def test_submission_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root)
            subprocess.run(["bash", str(REPO / "scripts/scc_submit.sh"), "smoke"],
                           env=env, check=True, capture_output=True)
            args = json.loads((root / "qsub.json").read_text())
            self.assertEqual(args[args.index("-P") + 1], "course-project")
            self.assertIn("gpus=1,gpu_c=7.0,gpu_memory=24G,h_rt=00:30:00", args)
            self.assertEqual(args[-1], str(REPO / "scripts/scc_job.sge"))

    def test_resume_targets_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = {**self.fixture(root), "SCC_MODE": "resume", "JOB_ID": "123"}
            subprocess.run(["bash", str(REPO / "scripts/scc_job.sge")], env=env,
                           check=True, capture_output=True)
            commands = [json.loads(line) for line in (root / "commands").read_text().splitlines()]
            self.assertEqual([args[2] for args in commands], ["check", "train", "export", "predict", "evaluate"])
            training = commands[1]
            self.assertIn("--resume", training)
            for flag in ("--baseline-updates", "--refiner-updates"):
                last = len(training) - 1 - training[::-1].index(flag)
                self.assertEqual(training[last + 1], "10")
            (root / "commands").unlink()
            failed = subprocess.run(["bash", str(REPO / "scripts/scc_job.sge")],
                                    env={**env, "FAIL_TRAIN": "1"}, capture_output=True)
            self.assertEqual(failed.returncode, 23)
            commands = [json.loads(line)[2] for line in (root / "commands").read_text().splitlines()]
            self.assertEqual(commands, ["check", "train"])


if __name__ == "__main__":
    unittest.main()
