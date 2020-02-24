import os
import argparse
import json


def read_json(file_name):
    with open(file_name) as json_data:
        return json.load(json_data)


DEFAULT_FILE_INPUT = './env1.json'
DEFAULT_KG = 0
DEFAULT_LAMBDA = 0.1
DEFAULT_OUTPUT = False
DEFAULT_OUTPUT_DIR = "./output"


def output(output_filename, data, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_filename)

    with open(output_file_path, 'w') as fp:
        json.dump(data, fp, indent=2)

    return output_file_path


def parse_args():

    parser = argparse.ArgumentParser(
        description='GUBS (Goals with Utility-Based Semantic) algorithm implementation.')

    parser.add_argument('--file', dest='file_input',
                        default=DEFAULT_FILE_INPUT,
                        help="Environment JSON file used as input (default: %s)" % DEFAULT_FILE_INPUT)
    parser.add_argument('--write_output', dest='output',
                        default=DEFAULT_OUTPUT,
                        action="store_true",
                        help="Defines whether or not to write the algorithm output to a file (default: %s)" % DEFAULT_OUTPUT)
    parser.add_argument('--c_max', dest='c_max', required=True,
                        type=int,
                        help="Maximum cost used in the algorithm")
    parser.add_argument('--kg', dest='kg',
                        default=DEFAULT_KG,
                        type=float,
                        help="Kg costant cost on goal states (default: %s)" % DEFAULT_KG)
    parser.add_argument('--lambda', dest='lamb',
                        default=DEFAULT_LAMBDA,
                        type=float,
                        help="Lambda risk factor (default: %s)" % DEFAULT_LAMBDA)
    return parser.parse_args()
