"""
set of functions to process 3D fruit model

"""
import trimesh
import shapely
import proc2d as pr2
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.spatial import distance


def obb_transform(mesh):
    mesh.apply_obb()
    return mesh


def if_trans_coordinate(mesh,v_axis):
    # If the position of the calyx curvature is y>0 in the plane passing through the origin cut by v_axis, t_conv is set
    # to 1 and returned.
    plane = [0,0,0]
    plane[v_axis] = 1
    max_retry = 5
    for c in range(max_retry):
        try:
            slice1 = mesh.section(plane_origin=mesh.centroid+c, plane_normal=plane)
            slice2, to_3D = slice1.to_planar()
            break
        except IndexError:
            pass
        except AttributeError:
            pass
    [[xmin,ymin],[xmax,ymax]] = slice2.bounds
    pol = pr2.Sections.select_largest_polygon(slice2)
    y_line = shapely.geometry.LineString([(0,ymin),(0,ymax)])
    pol_ring = shapely.geometry.LinearRing(pol.exterior.coords)
    y_inters = pol_ring.intersection(y_line)
    d3 = abs(y_inters[0].coords.xy[1][0]-ymin)
    d4 = abs(y_inters[1].coords.xy[1][0]-ymax)
    if y_inters[np.argmax([d3,d4])].coords.xy[1][0] >= 0: # y>=0にへた部があるなら
        t_conv = 1 # 1のときtrans_sectionsの順番を逆にする
    else:
        t_conv = 0
    return t_conv


def trans_slicing(mesh, t_axis, num_sect):
    origin = np.array([0, 0, 0])
    z_extents = mesh.bounds[:, t_axis]
    z_levels = np.linspace(*z_extents, num_sect)
    plane = [0, 0, 0]
    plane[t_axis] = 1

    sections = mesh.section_multiplane(plane_origin=origin,
                                       plane_normal=plane,
                                       heights=z_levels)
    return sections


def vert_slicing(mesh,t_axis,v_axis,num_sect):
    rotation_axis = t_axis
    init_point = v_axis

    u = np.zeros(3)
    u[rotation_axis] = 1
    plane = np.zeros(3)
    plane[init_point] = 1
    pi_levels = np.linspace(0,math.pi, num_sect)
    sections = []
    to3ds = []
    obb2ds = []
    for i in pi_levels:
        r = R(u,i)
        slice1 = mesh.section(plane_origin=mesh.centroid, plane_normal=plane@r)
        slice2, to_3D = slice1.to_planar()
        to3ds.append(to_3D)
        obb2ds.append(slice2.apply_obb())
        slice2.fill_gaps(100)
        sections.append(slice2)
    return sections, to3ds, obb2ds


def R(u, th):  # u軸θ回転の行列R (Rodriguesの回転公式)
    u = u.reshape([3, 1])
    return np.cos(th) * np.eye(3) + np.sin(th) * vec2skew(u) + (1 - np.cos(th)) * u @ u.T


def vec2skew(v):  # v∈R^3-->v_× (外積作用の行列)
    v = v.reshape([3, ])
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Hgps():
    """
    process horizontal groove points (numpy array object, nx2)
    """

    def __init__(self, mesh, hgrooves1, hgrooves2, t_axis, rotates, obb2ds, to3ds):
        self.hgps1 = hgrooves1
        self.hgps2 = hgrooves2
        self.mesh = mesh
        self.t_axis = t_axis
        self.rotates = rotates
        self.obb2ds = obb2ds
        self.to3ds = to3ds

    def hgps_process(self, vis=True):
        if self.hgps1 != 0 and self.hgps2 != 0:
            hgps3d = self.hgps_transform_in3d()
            area, rel_area = self.hgrooves_area(hgps3d, vis)
            vol, rel_vol = self.hgrooves_volume(hgps3d)
            dist = self.hgrooves_dist(hgps3d)
        else:
            area = 0
            rel_area = 0
            vol = 0
            rel_vol = 0
            dist = []
        return area, rel_area, vol, rel_vol, dist

    def hgps_transform_in3d(self):
        hgps3d = []
        for i in range(len(self.hgps1)):
            hgps = [np.append(self.hgps1[i], 1), np.append(self.hgps2[i], 1)]
            for hgp in hgps:
                if not np.any(np.isnan(hgp)):
                    tg_r = self.rotates[i].T @ hgp  # rotation back
                    tg_rr = np.linalg.inv(self.obb2ds[i]) @ tg_r  # obb back
                    tg_rrr = self.to3ds[i] @ np.append(tg_rr[:2], [0, 1])  # slicing back
                    hgps3d.append(tg_rrr[:3])
        return hgps3d

    def hgps_transform_in2d(self,hgps3d):
        # transform into the transverse origin plane
        # prepare transverse section at origin
        tplane = [0, 0, 0]
        tplane[self.t_axis] = 1
        t_sect1 = self.mesh.section(plane_origin=[0, 0, 0], plane_normal=tplane)
        t_sect2, to_3D = t_sect1.to_planar()

        col = np.zeros(len(hgps3d)) + 1
        hgs_stacked = np.column_stack((hgps3d, col))
        hgs_transformed = np.dot(to_3D, hgs_stacked.T).T[:, :3]  # transverse into the origin plane

        return hgs_transformed

    def hgrooves_area(self, hgps3d, vis):
        # prepare transverse section at origin
        tplane = [0, 0, 0]
        tplane[self.t_axis] = 1
        t_sect1 = self.mesh.section(plane_origin=[0, 0, 0], plane_normal=tplane)
        t_sect2, to_3D = t_sect1.to_planar()
        outer = t_sect2.polygons_full[0]
        outer = outer - outer.centroid
        outer = np.array(outer.exterior.coords.xy) - np.array(outer.centroid.xy)

        # transform into the transverse origin plane and clockwise sorting
        hgs_transformed = self.hgps_transform_in2d(hgps3d)
        hgs_transformed_xy = hgs_transformed[:, :2]
        idx = pr2.sort_XY_points_idx(hgs_transformed_xy)
        tgs_transformed_xy_sorted = hgs_transformed_xy[idx]
        center = np.array(shapely.geometry.Polygon(tgs_transformed_xy_sorted).centroid.xy)
        center = center.reshape(2)
        tgs_transformed_xy_sorted = tgs_transformed_xy_sorted - center

        # visualize
        if vis:
            fig = plt.figure(4, figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
            patch1 = patches.Polygon(outer.T, fc='gray', ec='k', alpha=0.2)
            ax.add_patch(patch1)
            patch2 = patches.Polygon(tgs_transformed_xy_sorted, fc='red', ec='k', alpha=0.2)
            ax.add_patch(patch2)
            ax.scatter(np.array(tgs_transformed_xy_sorted)[:, 0], np.array(tgs_transformed_xy_sorted)[:, 1], color="r",
                       alpha=0.5)

            fig.tight_layout()
        area = shapely.geometry.Polygon(tgs_transformed_xy_sorted).area
        rel_area = area/t_sect2.polygons_full[0].area
        return area, rel_area

    def hgrooves_volume(self,hgps3d):
        triangle_m = self.triangles(hgps3d)
        result = self.mesh_boolean(triangle_m)
        vol = result.volume
        rel_vol = result.volume / self.mesh.volume
        return vol, rel_vol

    def hgrooves_dist(self,hgps3d):
        dist = []
        centroid = np.average(hgps3d,axis=0)
        for tg in hgps3d:
            if not np.any(np.isnan(tg)):
                dist.append(distance.euclidean(tg, centroid))
        return dist

    def triangles(self,hgps3d):
        # clockwise sorting of tgps using the transformed xy points at origin plane
        tgs2d = self.hgps_transform_in2d(hgps3d)
        t_xy = [tuple(x) for x in tgs2d[:, :2]]
        idx = pr2.sort_XY_points_idx(t_xy)
        tgs3d_sorted = np.array(hgps3d)[idx]

        # prepare a mesh consisting of gps, centroid, and dummy_top
        u = np.zeros(3)
        u[self.t_axis] = 1
        centroid = np.average(tgs3d_sorted, axis=0)
        dummy_top = u * 100

        triangles = np.array([[centroid, tgs3d_sorted[0], tgs3d_sorted[-1]],
                              [tgs3d_sorted[-1], tgs3d_sorted[0], dummy_top]])
        for i in range(len(tgs3d_sorted) - 1):
            triangles = np.append(triangles, [[centroid, tgs3d_sorted[i + 1], tgs3d_sorted[i]]], axis=0)
            triangles = np.append(triangles, [[tgs3d_sorted[i], tgs3d_sorted[i + 1], dummy_top]], axis=0)

        triangle_m = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles))
        triangle_m.process(validate=True)  # to remove broken faces

        return triangle_m

    def mesh_boolean(self,triangle):
        # boolean operation
        result = trimesh.boolean.intersection([self.mesh, triangle])
        return result

    def hgps_vol_visualize(self,hgps3d):
        m = self.mesh.copy()
        m.visual.face_colors = [200, 200, 200, 22]
        scene = trimesh.Scene(m)
        triangle_m = self.triangles(hgps3d)
        result = self.mesh_boolean(triangle_m)
        result.visual.face_colors = [0, 255, 0, 2]
        scene.add_geometry(result)
        for hgp in hgps3d:
            if not np.any(np.isnan(hgp)):
                sp = trimesh.primitives.Sphere(radius=1, center=hgp)  # sphere prep
                sp.visual.face_colors = [255, 0, 0, 1]  # color
                scene.add_geometry(sp)
        return scene
