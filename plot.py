import argparse
import datetime
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utils import read_json
from mdp import get_actions

base_arrow = {
    'width': 0.04,
    'head_width': 0.3,
    'head_length': 0.2,
    'color': "white"
}

arrows = {
    'N': {
        'x': 0,
        'dx': 0,
        'y': 0.4,
        'dy': -0.6,
        **base_arrow
    },
    'S': {
        **base_arrow,
        'x': 0,
        'dx': 0,
        'y': -0.4,
        'dy': 0.6
    },
    'E': {
        **base_arrow,
        'x': -0.4,
        'dx': 0.6,
        'y': 0,
        'dy': 0
    },
    'W': {
        **base_arrow,
        'x': 0.4,
        'dx': -0.6,
        'y': 0,
        'dy': 0
    }
}


def plot_arrow(action, i, row_size, plt):
    arrow = arrows[action]
    plt.arrow(
        arrow['x'] + j % row_size,
        arrow['y'] + j // row_size,
        arrow['dx'], arrow['dy'],
        width=arrow['width'],
        head_width=arrow['head_width'], head_length=arrow['head_length'], color=arrow['color']
    )


def get_output_file_name(args, n_c):
    timestamp = datetime.datetime.now().timestamp()
    return f'{args.grid_width}_{args.grid_height}_{n_c}_{timestamp}.pdf'


env_file_input = './river7-10-40.json'

parser = argparse.ArgumentParser()
parser.add_argument('--env_file', dest='env_file',
                    default=env_file_input)

parser.add_argument('--results_file', dest='results_file', required=True)
parser.add_argument('--grid_width', dest='grid_width', required=True, type=int)
parser.add_argument('--grid_height', dest='grid_height',
                    required=True, type=int)
args = parser.parse_args()


output_dir = 'results'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

mdp_obj = read_json(args.env_file)
results_data = read_json(args.results_file)
V = np.array(results_data['V'])

S = list(mdp_obj.keys())
len_s = len(S)
A = get_actions(mdp_obj)

grid_height = args.grid_height
grid_width = args.grid_width

# read pi and V from output file
n_states = len(V)
n_c = 1 if not type(V[0]) == np.ndarray else len(V[0])
V = np.array(results_data['V']).T.reshape(n_c, 1,  n_states)
pi = np.array(results_data['pi']).T.reshape(n_c, 1,  n_states)

output_filename = get_output_file_name(args, n_c)
pp = PdfPages(output_dir + '/result' + output_filename)

for i_c in range(min(n_c, 10)):
    floor_v = V[i_c][0]
    plt.figure()

    n_columns = floor_v.shape[0]
    for j in range(n_columns):
        v = floor_v[j]
        row_size = n_columns / grid_height
        plt.text(j % row_size, j // row_size,
                 "{0:.2f}".format(v))
        action = pi[i_c][0][j]
        if action:
            plot_arrow(action, j, row_size, plt)

    plt.imshow(floor_v.reshape(grid_height, int(n_states / grid_height)))
    plt.title('C: ' + str(i_c))
    plt.savefig(pp, format="pdf")

pp.close()
