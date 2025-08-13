import os
import argparse
import dotenv
dotenv.load_dotenv()
from tqdm.contrib.concurrent import process_map

# from bloxnet.pipelines.parallel_whole_structure import generate_structure
from bloxnet.pipelines.whole_structure import generate_structure
from bloxnet.pybullet.place_blocks_in_json import init_pybullet
from bloxnet.utils.utils import load_from_json, write_error

# structure_name = "Bridge"

def make_pybullet(x):
    to_build, num_structures,json_file = x
    dotenv.load_dotenv()
    blockset = load_from_json(json_file)

    init_pybullet(gui=False)
    assembly = generate_structure(to_build, blockset, iter=num_structures)


    return assembly


def main_run_pipeline_single_obj_parallel(to_build, num_structures=10, max_workers=10,json_file='blocksets/some_cuboid_blocks.json'):
    # Ensure this is only run in the main process
    assert max_workers>0, 'max workers must be greater than 0'

    assemblies = process_map(
        make_pybullet,
        zip([to_build] * num_structures, range(num_structures),[json_file] * num_structures),
        max_workers=max_workers)

    assemblies = list(filter(lambda x: x is not None, assemblies))

    init_pybullet(gui=False)
    return assemblies

def store_assembly(assembly, seq_iter):
    # assembly = run_perturbation_analysis(assembly)
    if seq_iter != 0:
        save_dir = f"{assembly.structure_directory}/replan_assembly_{seq_iter}"
    else:
        save_dir =  f"{assembly.structure_directory}/initial_assembly_{seq_iter}"

    assembly.human_friendly_save(save_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pipeline for single object in parallel')
    parser.add_argument('structure_name', type=str, help='Name of the structure to build')
    parser.add_argument('--num_structures', type=int, default=10,
                      help='Number of structures to generate (default: 10)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of workers for parallel generation (default: 10)') 
    args = parser.parse_args()
    main_run_pipeline_single_obj_parallel(args.structure_name, args.num_structures, args.num_workers)
