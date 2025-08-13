import dotenv
dotenv.load_dotenv()

from prompting_and_structure.pipelines.whole_structure import generate_structure
from prompting_and_structure.pybullet.place_blocks_in_json import init_pybullet
from prompting_and_structure.utils.file_and_parse_utils import load_from_json

def generate_initial_assembly(to_build, json_file):
    dotenv.load_dotenv()
    blockset = load_from_json(json_file)

    init_pybullet(gui=False)
    assembly = generate_structure(to_build, blockset, iter=0)
    return assembly

def store_assembly(assembly, seq_iter):
    if seq_iter != 0:
        save_dir = f"{assembly.structure_directory}/replan_assembly_{seq_iter}"
    else:
        save_dir =  f"{assembly.structure_directory}/initial_assembly_{seq_iter}"

    assembly.human_friendly_save(save_dir)
    return

if __name__ == "__main__":
    structure_name = "bridge"
    json_file = "blocksets/bridge_blocks.json"

    generate_initial_assembly(structure_name, json_file)
