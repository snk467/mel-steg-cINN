import yaml
import munch

LOCAL_CONFIG_FILE="config_local.yml"

def load():
    config_file = open(LOCAL_CONFIG_FILE, "r")
    yaml_config = yaml.safe_load(config_file)
    config = munch.munchify(yaml_config)
    return config