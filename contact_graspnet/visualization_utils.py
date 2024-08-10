import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os
import shutil
import open3d as o3d
from scipy.spatial.transform import Rotation

import grasps.aograsp.contact_graspnet.contact_graspnet.mesh_utils as mesh_utils
mlab.options.offscreen = True
# mlab.options.backend = 'envisage'

def viz_proposals_mlab(
    cgn_grasps,
    grasp_scores,
    pts,
    heatmap,
    save_path=None,
    draw_ref_frame=False,
    save_as_mp4=False,
    draw_all_grasps=True,
    bgcolor=None,
    pcd_rgb=None,
    highlight_top_k=10,
    highlight_id=None,
    pc_mode="2dsquare",
    pc_scale_factor=0.0018,
    pc_opacity=1.0,
    gripper_openings=None,
):

    fig = mlab.figure('Pred Grasps', size=(1024, 1024), bgcolor=bgcolor)
    mlab.view(azimuth=180, elevation=180, distance=2, roll=0) # rotated data

    # Scale heatmap for visualization
    heatmap = scale_to_0_1(heatmap)

    if pcd_rgb is not None:
        draw_pc_with_colors(
            pts,
            pcd_rgb,
            use_heatmap=False,
            mode=pc_mode,
            scale_factor=pc_scale_factor,
            opacity=pc_opacity,
        )
    else:
        draw_pc_with_colors(
            pts,
            heatmap,
            use_heatmap=True,
            mode=pc_mode,
            scale_factor=pc_scale_factor,
            opacity=pc_opacity,
        )
    #draw_pc_with_colors(pts, heatmap, use_heatmap=True, mode="sphere", scale_factor=0.01)

    #colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    #colors2 = {k:cm2(0.5*np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}
    
    if gripper_openings is None:
        gripper_width=0.08
        gripper_openings = np.ones(len(cgn_grasps))*gripper_width

    # Sort grasps from high to low scores
    cgn_grasp_tuples = [(cgn_grasps[i], grasp_scores[i], gripper_openings[i]) for i in range(len(grasp_scores))]
    cgn_grasp_tuples.sort(key=lambda data: -data[1])
    sorted_cgn_grasps = [grasp_tuple[0] for grasp_tuple in cgn_grasp_tuples]
    sorted_grasp_scores = [grasp_tuple[1] for grasp_tuple in cgn_grasp_tuples]
    sorted_gripper_openings = [grasp_tuple[2] for grasp_tuple in cgn_grasp_tuples]

    cmap = matplotlib.cm.get_cmap("RdYlGn")
    colors = []
    for score in sorted_grasp_scores:
        colors.append(cmap(score)[:3])

    # Draw other grasps in color based on score (thin line)
    if draw_all_grasps:
        draw_grasps_ours(
            cgn_grasps,
            np.eye(4),
            color=(0,0,0), # black
            #colors=colors[top_k:], # scores
            gripper_openings=sorted_gripper_openings,
            tube_radius=0.0004,
        )

    if highlight_top_k is not None:
        # Draw top_k grasps in thick green
        draw_grasps_ours(
            sorted_cgn_grasps[:highlight_top_k],
            np.eye(4),
            color=(51/255, 204/255, 51/255), # green
            #color=(0.6, 0.1, 1),
            tube_radius=0.0035,
            gripper_openings=sorted_gripper_openings
        )
    if highlight_id is not None:
        draw_grasps_ours(
            [sorted_cgn_grasps[highlight_id]],
            np.eye(4),
            color=(51/255, 204/255, 51/255), # green
            #color=(0.6, 0.1, 1),
            #tube_radius=0.01,
            tube_radius=0.0035,
            gripper_openings=sorted_gripper_openings
        )

    if draw_ref_frame:
        plot_coordinates(np.zeros(3,),np.eye(3,3))

    if save_path is not None:
        if not save_as_mp4:
            f = mlab.gcf()
            f.scene._lift()
            mlab.savefig(filename=save_path, figure=f)
            mlab.close()
            im = Image.open(save_path)
            im = im.rotate(90, expand=True)
            im.save(save_path)
            print('Saved image to ', save_path)
        else:
            # Define the number of frames and angles for the animation
            degree_delta = 5
            # Left
            angle_list = [] # list of [a, e, r] angles
            start_ang = 120
            #start_ang = 90
            for e_angle in np.arange(start_ang, 180, degree_delta):
                angle_list.append([180, e_angle, 180])
            angle_list.append([270, 180, 180]) # MIDDLE
            # Right
            for e_angle in np.arange(180-degree_delta, start_ang-degree_delta, -degree_delta):
                angle_list.append([360, e_angle, 180])

            # Overwrite tmp directory to store frames for gif
            frames_dir = 'point_cloud_frames'
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)

            # Rotate the camera around the point cloud and save each frame
            k = 0
            for aer in angle_list:
                a_angle = aer[0]
                e_angle = aer[1]
                r_angle = aer[2]
                mlab.view(elevation=e_angle, azimuth=a_angle, roll=r_angle, distance=2)
                mlab.draw()
                frame_path = os.path.join(frames_dir, f'frame_{k:03d}.png')
                mlab.savefig(frame_path)
                k += 1

            # Convert the frames into a movie using an external tool
            if highlight_id is not None:
                movie_save_path = os.path.splitext(save_path)[0] + "_single.mp4"
            else:
                movie_save_path = os.path.splitext(save_path)[0] + ".mp4"
            os.system(f'ffmpeg -framerate 10 -i {frames_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {movie_save_path}')
            print('Saved mp4 to ', movie_save_path)

    else: 
        mlab.show()

def plot_mesh(mesh, cam_trafo=np.eye(4), mesh_pose=np.eye(4)):
    """
    Plots mesh in mesh_pose from 

    Arguments:
        mesh {trimesh.base.Trimesh} -- input mesh, e.g. gripper

    Keyword Arguments:
        cam_trafo {np.ndarray} -- 4x4 transformation from world to camera coords (default: {np.eye(4)})
        mesh_pose {np.ndarray} -- 4x4 transformation from mesh to world coords (default: {np.eye(4)})
    """
    
    homog_mesh_vert = np.pad(mesh.vertices, (0, 1), 'constant', constant_values=(0, 1))
    mesh_cam = homog_mesh_vert.dot(mesh_pose.T).dot(cam_trafo.T)[:,:3]
    mlab.triangular_mesh(mesh_cam[:, 0],
                         mesh_cam[:, 1],
                         mesh_cam[:, 2],
                         mesh.faces,
                         colormap='Blues',
                         opacity=0.5)

def plot_coordinates(t,r, tube_radius=0.005):
    """
    plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """
    mlab.plot3d([t[0],t[0]+0.2*r[0,0]], [t[1],t[1]+0.2*r[1,0]], [t[2],t[2]+0.2*r[2,0]], color=(1,0,0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0],t[0]+0.2*r[0,1]], [t[1],t[1]+0.2*r[1,1]], [t[2],t[2]+0.2*r[2,1]], color=(0,1,0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0],t[0]+0.2*r[0,2]], [t[1],t[1]+0.2*r[1,2]], [t[2],t[2]+0.2*r[2,2]], color=(0,0,1), tube_radius=tube_radius, opacity=1)
                
def show_image(rgb, segmap):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.ion()
    plt.show()
    
    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)   
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)

def visualize_grasps(full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None, gripper_width=0.08, save=False, filename=None, use_heatmap=False):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions. 
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print('Visualizing...takes time')
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')
   
    fig = mlab.figure('Pred Grasps', size=(1024, 1024))
    # mlab.view(azimuth=0, elevation=-90, distance=0.2, roll=90) # original data
    mlab.view(azimuth=180, elevation=180, distance=2, roll=0) # rotated data
    draw_pc_with_colors(full_pc, pc_colors, use_heatmap=use_heatmap)
    colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k:cm2(0.5*np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}
    
    if plot_opencv_cam:
        plot_coordinates(np.zeros(3,),np.eye(3,3))
    
    for i,k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k]))*gripper_width if gripper_openings is None else gripper_openings[k]
            if len(pred_grasps_cam) > 1:
                draw_grasps(pred_grasps_cam[k], np.eye(4), color=colors[i], gripper_openings=gripper_openings_k)    
                draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), color=colors2[k],gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0025)    
            else:
                colors3 = [cm2(0.5*score)[:3] for score in scores[k]]
                draw_grasps(pred_grasps_cam[k], np.eye(4), colors=colors3, gripper_openings=gripper_openings_k)
                draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), color=(0.6,0.1,1),gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0035)   
    if save:
        f = mlab.gcf()
        f.scene._lift()
        mlab.savefig(filename=filename, figure=f)
        mlab.close()
        im = Image.open(filename)
        im = im.rotate(90, expand=True)
        im.save(filename)
        print('Saved image to ', filename)
    else: 
        mlab.show()

def draw_pc_with_colors(
    pc,
    pc_colors=None,
    single_color=(0.3,0.3,0.3),
    mode='2dsquare',
    scale_factor=0.0018,
    use_heatmap=False,
    opacity=1.0
):
    """
    Draws colored point clouds

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=single_color, scale_factor=scale_factor, mode=mode)
    elif use_heatmap:
        mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            pc_colors,
            colormap='RdYlGn',
            mode=mode,
            scale_factor=scale_factor,
            scale_mode="none",
            vmax=1.0,
            vmin=0.0,
            transparent=False,
            opacity=opacity,
            resolution=32,
        )
    else:
        #create direct grid as 256**3 x 4 array 
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack((xl[0].reshape(1, 256**3),
                                xl[1].reshape(1, 256**3),
                                xl[2].reshape(1, 256**3),
                                255 * np.ones((1, 256**3)))).T
            return lut.astype('int32')
        
        scalars = pc_colors[:,0]*256**2 + pc_colors[:,1]*256 + pc_colors[:,2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            scalars,
            mode=mode,
            scale_factor=scale_factor,
            scale_mode="none",
            transparent=False,
            opacity=opacity,
            #resolution=32,
        )
        points_mlab.glyph.scale_mode = 'scale_by_vector'
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut

def draw_grasps(grasps, cam_pose, gripper_openings, color=(0,1.,0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])
        
    all_pts = []
    connections = []
    index = 0
    N = 7
    for i,(g,g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening/2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N

        if colors is not None:
            # Draw grasps individually
            color = colors[i]
            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

    if colors is None:
        # speeds up plot3d because only one vtk object
        all_pts = np.vstack(all_pts)
        connections = np.vstack(connections)
        src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
        src.mlab_source.dataset.lines = connections
        src.update()
        lines =mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
        mlab.pipeline.surface(lines, color=color, opacity=1.0)
    

def draw_grasps_ours(grasps, cam_pose, gripper_openings, color=(0,1.,0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
    """
    Draws wireframe grasps for robotiq schematic from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    mid_point[2] -= 0.02
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], gripper_control_points[1], mid_point, gripper_control_points[2], gripper_control_points[4]])

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])
        
    all_pts = []
    connections = []
    index = 0
    N = 8
    for i,(g,g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        finger_idx = [2,3,4,6,7]
        gripper_control_points_closed[finger_idx,0] = np.sign(grasp_line_plot[finger_idx,0]) * g_opening/2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((8, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N

        if colors is not None:
            # Draw grasps individually
            color = colors[i]

            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

    if colors is None:
        # speeds up plot3d because only one vtk object
        all_pts = np.vstack(all_pts)
        connections = np.vstack(connections)
        src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
        src.mlab_source.dataset.lines = connections
        src.update()
        lines =mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
        mlab.pipeline.surface(lines, color=color, opacity=1.0)
    

def get_o3d_pts(pts, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def get_eef_line_set_for_o3d_viz(eef_pos_list, eef_quat_list, highlight_top_k=None):
    # Get base gripper points
    g_opening = 0.07
    gripper = mesh_utils.create_gripper("panda")
    gripper_control_points = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array(
        [
            np.zeros((3,)),
            mid_point,
            gripper_control_points[1],
            gripper_control_points[3],
            gripper_control_points[1],
            gripper_control_points[2],
            gripper_control_points[4],
        ]
    )
    gripper_control_points_base = grasp_line_plot.copy()
    gripper_control_points_base[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
    # Need to rotate base points, our gripper frame is different
    # ContactGraspNet
    r = Rotation.from_euler("z", 90, degrees=True)
    gripper_control_points_base = r.apply(gripper_control_points_base)

    # Compute gripper viz pts based on eef_pos and eef_quat
    line_set_list = []
    for i in range(len(eef_pos_list)):
        eef_pos = eef_pos_list[i]
        eef_quat = eef_quat_list[i]

        gripper_control_points = gripper_control_points_base.copy()
        g = np.zeros((4, 4))
        rot = Rotation.from_quat(eef_quat).as_matrix()
        g[:3, :3] = rot
        g[:3, 3] = eef_pos.T
        g[3, 3] = 1
        z = gripper_control_points[-1, -1]
        gripper_control_points[:, -1] -= z
        gripper_control_points[[1], -1] -= 0.02
        pts = np.matmul(gripper_control_points, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)

        lines = [[0, 1], [2, 3], [1, 4], [1, 5], [5, 6]]
        if highlight_top_k is not None:
            if i < highlight_top_k:
                # Draw grasp in green
                colors = [[0,1,0] for i in range(len(lines))]
            else:
                colors = [[0,0,0] for i in range(len(lines))]
        else:
            colors = [[0,0,0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)

    return line_set_list


def viz_pts_and_eef_o3d(
    pts_pcd,
    eef_pos_list,
    eef_quat_list,
    heatmap_labels=None,
    save_path=None,
    frame="world",
    draw_frame=False,
    cam_frame_x_front=False,
    highlight_top_k=None,
    pcd_rgb=None
):
    """
    Plot eef in o3d visualization, with point cloud, at positions and
    orientations specified in eef_pos_list and eef_quat_list
    pts_pcd, eef_pos_list, and eef_quat_list need to be in same frame
    """
    print('o3ding...')
    pcd = get_o3d_pts(pts_pcd)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        if heatmap_labels is not None:
            # Scale heatmap for visualization
            heatmap_labels = scale_to_0_1(heatmap_labels)

            cmap = matplotlib.cm.get_cmap("RdYlGn")
            colors = cmap(np.squeeze(heatmap_labels))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get line_set for drawing eef in o3d
    line_set_list = get_eef_line_set_for_o3d_viz(
        eef_pos_list, eef_quat_list, highlight_top_k=highlight_top_k,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    for line_set in line_set_list:
        vis.add_geometry(line_set)

    # Draw ref frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)

    # Move camera
    if frame == "camera":
        # If visualizing in camera frame, view pcd from scene view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        if cam_frame_x_front:
            R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
            H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        # If world frame, place camera accordingly to face object front
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, -1] = 1
        R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
        H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()

def scale_to_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
