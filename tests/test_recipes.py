import logging

from launcher.nemo.constants import ROOT_DIR

from .test_utils import get_launcher_run_script_paths

logger = logging.getLogger(__name__)

RUN_SCRIPT_PATHS = get_launcher_run_script_paths()


def test_config_for_run_script_exists():
    RECIPES_DIR = ROOT_DIR / "recipes_collection/recipes"

    for run_script_path in RUN_SCRIPT_PATHS:
        with open(run_script_path, "r") as fd:
            for line in fd:
                # this line defines the Yaml configuration file
                #  example: recipes=training/llama/hf_llama3_2_90b_seq8k_gpu_p5x32_pretrain
                if "recipes=" in line:
                    # clean up line
                    line = line.replace(" \\", "")  # remove shell line continuation marker
                    line = line.strip()

                    _, config_path_str = line.split("=")
                    config_path = RECIPES_DIR / (config_path_str + ".yaml")  # append .yaml

                    logger.info(
                        f"\nlauncher file: {run_script_path.relative_to(ROOT_DIR)}"
                        f"\nconfig file: {config_path.relative_to(ROOT_DIR)}"
                        "\n"
                    )

                    assert config_path.exists()
