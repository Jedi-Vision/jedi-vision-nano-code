from jv import Driver
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
    type=str,
    default="config/default.yaml",
    help="Configuration file containing arguments for Driver."
)
parser.add_argument(
    "-a", "--args",
    type=str,
    action="store",
    nargs='*',
    help="Arguments to replace any config field in 'key:value' style."
)
args = parser.parse_args()

extra_args = {}
if args.args:
    for arg in args.args:
        k, v = arg.split(":")
        extra_args[k] = v

with open(args.config) as f:
    config = yaml.safe_load(f)

driver_kwargs = config['driver']
object_kwargs = config['object']
depth_kwargs = config['depth']

for key, value in extra_args.items():
    if key in driver_kwargs:
        driver_kwargs[key] = value
    elif key in object_kwargs:
        object_kwargs[key] = value
    elif key in depth_kwargs:
        depth_kwargs[key] = value

driver = Driver(**driver_kwargs)
driver.run(object_kwargs, depth_kwargs)
