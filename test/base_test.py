from rl_scs.utils.package_utils import get_package_root


class BaseTest:
    
    configs = [
        "test_config.yml",
        "randomized_config_5.yml",
        "randomized_config_10.yml",
        "mirrored_config_5.yml",
        "solo_soldier_config_5.yml",
        "unbalanced_config_5.yml"
    ]
    
    @staticmethod
    def get_config_path(config_name):
        package_root = get_package_root()
        return str(package_root / "example_configurations" / config_name)
