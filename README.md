<h1 align="center">
    Iterative Design for Robotic Assembly (IDfRA)
</h1>

<p align="center">
  <img alt="giraffe_vid" src="assembly_evolution_video/Giraffe_Assembly_Evolution.gif" width="400" />
</p>

# Setup

## Installation
```
git clone 
conda env create -f environment.yaml
conda activate idfra
```

## OpenAI API Key
IDfRA uses the ChatGPT via [OpenAI API](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key): create a file named ```.env``` in the root directory of the repository and include ```OPENAI_API_KEY=[your api key]```


# Running IDfRA

## Repository Structure
### Directories
- ```prompting_and_structure/```: extensively modified code from BloxNet for prompting GPT, structural functions
- ```blocksets/```: all blocksets used for assembly generation
### Scripts
- ```run_idfra_pipeline.py```: runs end-to-end IDfRA pipeline in simulation 
- ```pick_and_place_env.py```: custom simulation environment class 
- ```robot_suction.py```: custom simulation environment class 
- ```utils.py```: misc utils for dimension switch, plots, pictures 
- ```vlm_resemblance_evaluation.py```: code for running VLM-based quantitative assessment

## Generating Assembly Structures
To iteratively generate assemblies using IDfRA, run:
```
python run_idfra_pipeline.py
```
