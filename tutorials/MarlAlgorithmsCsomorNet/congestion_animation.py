import os

import pandas as pd
import random

from collections import Counter

import janux as jx

if __name__ == "__main__":
    
    ##################### PARAMS ############################

    network_name = "csomor"
    
    # File paths
    nod_file_path = f"../../RouteRL/network_and_config/{network_name}/{network_name}.nod.xml"
    edg_file_path = f"../../RouteRL/network_and_config/{network_name}/{network_name}.edg.xml"

    num_frames = 20
    interval = 10
    frame_duration = 500
    
    read_routes_from = f"training_records/paths.csv"
    save_figs_path = f"plots/"
    save_frames_path = f"plots/frames/"
    save_gif_to = f"plots/animation.gif"

    ########################################################
    
    ##################### Create mock edge attributes ############################
    os.makedirs(save_figs_path, exist_ok=True)
    routes = pd.read_csv(read_routes_from)
    routes = routes["path"].values

    # Put all edges in a single list
    all_edges = list()
    for route in routes:
        all_edges += route.split(" ")
        
    all_edges = list(set(all_edges))
    
    i=100
    max_flow, min_flow = -1, 1000000
    while i<=2000:
        detector_df = pd.read_csv(f"training_records/detector/detector_ep{i}.csv")
        flows = detector_df["flow"].values
        max_flow = max(max_flow, max(flows))
        min_flow = min(min_flow, min(flows))
        i += interval
    
    congestion_dicts = list()
    i=1
    while i<=2000:
        detector_df = pd.read_csv(f"training_records/detector/detector_ep{i}.csv")
        flows = detector_df["flow"].values
        normalized_flows = (flows - min_flow) / (max_flow - min_flow)
        congestions = {det_id: flow for det_id, flow in zip(detector_df["detid"], normalized_flows)}
        congestion_dict = {edge: congestions.get(edge, 0) for edge in all_edges}
        congestion_dicts.append(congestion_dict)
        i += interval
        
    #################### Animate ############################ 
    
    jx.animate_edge_attributes(nod_file_path,
                            edg_file_path,
                            congestion_dicts,
                            save_frames_dir=save_frames_path,
                            save_gif_to=save_gif_to,
                            frame_duration=frame_duration)
