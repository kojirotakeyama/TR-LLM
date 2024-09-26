# TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction

We present a multi-modal human action prediction framework that incorporates both an LLM and human trajectories. The core idea is to integrate two different perspectives—physical and semantic factors—through an object-based action prediction framework to reduce uncertainties and enhance action prediction accuracy. This site provides codes and dataset to test our method.

<!--
<center>
 <img src="./Overview.png" alt="Overview" width="800">
</center>
-->
![Overview](./Overview.png)


## Quick start
Quick visualization of the trajectories contained in LocoVR is done by the following instructions.

1. Download github repo

3. Download map images
   Download the foloder "vrlocomotion_models_000" from following link and unzip it, then place it in the top of "main" folder.
   [Download model](https://drive.google.com/drive/folders/1A9NCngHYVbUDx3M7P638edZfMieJlayY?usp=sharing)
     
4. Install the packages in requirements.txt (python==3.8.1, cuda12.1):
```
pip install -r requirements.txt
```
5. Calculate
   Run vis_trajectory.py, you will get time-sereis trajectory images of the specified scenes.
```
python ./visualize_trajectory/vis_traj.py
```
Tips: 
- To change the scene of visualization, edit "scene_id" in the config.yaml. You can choose multiple scenes with a list style.
- To change the map type, edit "map type" in the config.yaml.
- To change the type of visualizing trajectory, edit "type_viz" to "waist" or "body".

## Map generation
All the 2D maps provided in the "Quick start" are generated basd on HM3DSem datasets.
Following instruction will help you generating the maps.

1. Download Habitat Dataset
Download Habitat Dataset from the following link and place it on the directory you specified on the config.yaml.
To generate the maps, download HM3DSem datasets (hm3d-train-semantic-annots-v0.2.tar/hm3d-val-semantic-annots-v0.2.tar) from the following link.
- [HM3DSem](https://github.com/matterport/habitat-matterport-3dresearch/tree/main)

If you just want to test the map generation code, you can do it with small sample data: hm3d-example-semantic-annots-v0.2.tar

2. Generate maps
  You can generate 2D maps from the HM3DSem Dataset through running generate_map.py as follows.
```
python ./Generate_map/generate_map.py
```
Tips: 
- To change the map type (binary, height, semantic, texture), modify "map_type" in the config.yaml.
- If you need to generate photo realistic texture map, download HM3D (including .obj) from the following link.
  - [HM3D](https://matterport.com/partners/facebook)

## Evaluation codes for NeurIPS2024
Evaluation code and data for NeurIPS2024 is available at [NeurIPS2024](https://anonymous.4open.science/r/NeurIPS2024-1FD1/README.md)

## Citation
If you find this repo useful for your research, please consider citing:

## License
This project is licensed under the MIT License
