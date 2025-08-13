from pick_and_place_env import PickPlaceEnv

if __name__ == "__main__":
        
        # Run configuration
        gui = True
        json_file = "blocksets/giraffe_blocks.json"
        max_seq_iterations = 2

        # Initialise environment
        env = PickPlaceEnv(gui=gui,json_file=json_file,max_seq_iterations=max_seq_iterations)
        obs = env.reset()

        # Run IDfRA pipeline
        env.run_idfra_pipeline()