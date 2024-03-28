import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

from cgn_grasp_proposer import CGN_Grasp_Proposer
import tensorflow.compat.v1 as tf

"""
Script to read from a heatmap file run CGN it 
"""

def main(args):
    if args.display is not None:
        # Set env variable DIPSLAY to user-specified value
        os.environ["DISPLAY"] = args.display

    cgn = CGN_Grasp_Proposer()

    # Make save directories
    data_dir = os.path.dirname(os.path.dirname(args.heatmap_file))
    prop_save_dir = os.path.join(data_dir, "grasp_proposals")
    img_save_dir = os.path.join(data_dir, "grasp_proposals_img")
    if not os.path.exists(prop_save_dir): os.makedirs(prop_save_dir)
    if not os.path.exists(img_save_dir): os.makedirs(img_save_dir)

    cgn.propose_grasp_from_heatmap_file(
        args.heatmap_file,
        prop_save_dir = prop_save_dir,
        img_save_dir = img_save_dir,
        heatmap_source = args.heatmap_source,
        viz_o3d = args.viz_o3d,
        viz_save_as_mp4=args.viz_save_as_mp4,
        viz_all_grasps=args.viz_all_grasps,
        viz_id=args.viz_id,
        viz_top_k=args.viz_top_k,
        viz_heatmap=args.viz_heatmap,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("heatmap_file", type=str, help="Path to heatmap .npz file")
    parser.add_argument("--heatmap_source", default="ours", choices=["ours", "gt", "cgn"], help="What point scores to use")
    parser.add_argument(
        "--display",
        type=str,
        choices=[":1", ":2", ":3"],
        help="Display number",
    )
    parser.add_argument("--viz_o3d", action="store_true", help="Visualize proposals in o3d visualizer")
    parser.add_argument("--viz_heatmap", action="store_true", help="Visualize point colors as heatmap")
    parser.add_argument("--viz_all_grasps", action="store_true", help="Visualize all grasps w/ thin black line")
    parser.add_argument("--viz_save_as_mp4", action="store_true", help="Save proposals gif")
    parser.add_argument("--viz_id", type=int, help="Grasp ID to highlight w/ green line")
    parser.add_argument("--viz_top_k", type=int, help="Top-k grasps to highlight w/ green line")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
