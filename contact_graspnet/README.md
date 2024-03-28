# CGN for AO-Grasp

Code to generate orientation proposals with CGN for heatmaps predicted by AO-Grasp. 

## Installation 

### Create conda env

Create a conda env for CGN with our custom yaml:

```
conda env create -f aograsp_cgn_environment.yml -n cgn
```

### Download models

Download trained models from [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing) and copy them into the `contact_graspnet/checkpoints/` folder.

### Test installation

Generate and save proposals for test data. Will see an image of top 10 proposals saved to `contact_graspnet/test_data/proposals-cgn-p-heatmap_ours_img/data_000.png` and proposals saved to `contact_graspnet/test_data/proposals-cgn-p-heatmap_ours/data_000.npz`.

From within the `contact_graspnet/` repository, run the following command. Note: You will need to have either a physical or remote display (vncserver) running for the visualization functions to work.

```
python contact_graspnet/run_cgn_on_heatmap_file.py contact_graspnet/test_data/point_score/data_000.npz --viz_top_k 10
```

## Usage

### Generate proposals for a single heatmap file

Generate proposals for single heatmap file `.../test_output/point_score/saved_heatmap.npz` (saved from AO-Grasp).

```
python contact_graspnet/run_cgn_on_heatmap_file.py <path/to/heatmap.npz> <--VISUALIZATION OPTIONS SEE BELOW>
```

#### Visualization options

- `--viz_o3d` (store_true): If this flag is used, grasp proposals will be visualized in an open3d visualizer window.
- `--viz_all_grasps` (store_true): If this flag is used, all top 200 grasps (minus duplicates) will be visualized with a thin black line.
- `--viz_save_as_mp4`(store_true): If this flag is used, grasp proposal visualization will be saved as a gif. If flag is not used, a `.png` image will be saved.
- `--viz_id` (int): If specified, highlight grasp with id `viz_id` with thick green line.
- `--viz_top_k` (int): If specified, highlight `viz_top_k` highest-scoring grasps with thick green lines.
