"""
set of functions to process fruit slice data

"""
import numpy as np
import trimesh
import copy
import shapely
import math
import smallestenclosingcircle
from polylabel import polylabel
from descartes.patch import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib import patches
import curvaturenumeric as cn
from pyclustering.cluster import gmeans
import itertools
from shapely.strtree import STRtree
import operator
from functools import reduce


def get_lw(slice):
    # Length/Width
    bb = slice.polygons_full[0].bounds
    d = (bb[3] - bb[1]) / (bb[2] - bb[0])
    return d


def findDeepest(pol_r):
    # pol_r = pr_1
    pr_R = pol_r[pol_r[:, 0] > 0]
    try:
        pr_r_sorted = pr_R[np.argsort(pr_R[:, 0])]
        xmax = max(pol_r[:, 0])
        xmin = min(pol_r[:, 0])
        xbin = xmax / 20
        point_ref = pr_r_sorted[0]
        delta_y = 0
        for j in range(int(len(pr_r_sorted) / 2)):
            point1 = pr_r_sorted[j]
            for k in np.arange(j + 1, len(pr_r_sorted) - j - 1):
                point2 = pr_r_sorted[k]
                if point2[0] - point1[0] >= xbin:
                    delta_y = point2[1] - point1[1]
                    break
            if delta_y > 0:
                point_ref = point2
                break
        pr_R = pr_R[pr_R[:, 0] >= point_ref[0]]
        pr_Rmax = pr_R[np.argmax(pr_R[:, 1])]
    except ValueError:
        pr_Rmax = np.array([np.nan, np.nan])
    except IndexError:
        pr_Rmax = np.array([np.nan, np.nan])

    pr_L = pol_r[pol_r[:, 0] <= 0]
    try:
        pr_l_sorted = pr_L[np.argsort(pr_L[:, 0])[::-1]]
        xmax = max(pol_r[:, 0])
        xmin = min(pol_r[:, 0])
        xbin = abs(xmin / 20)
        point_ref = pr_l_sorted[0]
        delta_y = 0
        for j in range(int(len(pr_l_sorted) / 2)):
            point1 = pr_l_sorted[j]
            for k in np.arange(j + 1, len(pr_l_sorted) - j - 1):
                point2 = pr_l_sorted[k]
                if point1[0] - point2[0] >= xbin:
                    delta_y = point2[1] - point1[1]
                    break
            if delta_y > 0:
                point_ref = point2
                break
        pr_L = pr_L[pr_L[:, 0] <= point_ref[0]]
        pr_Lmax = pr_L[np.argmax(pr_L[:, 1])]
    except ValueError:
        pr_Lmax = np.array([np.nan, np.nan])
    except IndexError:
        pr_Lmax = np.array([np.nan, np.nan])

    if not np.any(np.isnan(np.array([pr_Rmax, pr_Lmax]))):
        pr_in_region = pol_r[(pr_Lmax[0] <= pol_r[:, 0]) & (pol_r[:, 0] <= pr_Rmax[0])]
        point = pr_in_region[np.argmin(pr_in_region[:, 1])]
    else:
        point = np.array([np.nan, np.nan])
    return point


def rotate_origin(xy, radians):
    """rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def init_from_miny(coord):
    # 座標の点列をyが最も小さい場所から始めるように順番を入れ替える
    ymin = np.argmin(coord, axis=0)[1]
    new_coord = np.concatenate([coord[ymin:], coord[:ymin]])
    return new_coord


def anticlockwise(coord):
    # 座標の点列を反時計回りに順番を入れ替える
    direction = coord[0][0] * coord[3][1] - coord[0][1] * coord[3][0]
    if direction < 0:  # 時計回り
        coord = np.flipud(coord)
    return coord


def rotate90(sect):
    # rotate path2D object
    return sect.apply_transform(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))


def rotate180(sect):
    return sect.apply_transform(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))


def rotate270(sect):
    return sect.apply_transform(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))


def sort_XY_points_idx(array_2d):
    # 参考：https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates/51075469
    # 点群位置を時計回りにソート
    t_xy = [tuple(x) for x in array_2d]
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), t_xy), [len(t_xy)] * 2))
    idx = np.argsort(
        list(map(lambda x: (-135 - math.degrees(math.atan2(*operator.sub(x, np.array(center))))) % 360, t_xy)))
    return idx


class TransSect(shapely.geometry.polygon.Polygon):
    """
    process a transverse section (shapely.geometry.polygon.Polygon object)
    """

    def __init__(self, pol):
        self.pol = pol
        super().__init__()

    def convexity_defect_area(self):
        # calculate convexity defect area of the section
        defect = self.pol.convex_hull.area - self.pol.area
        return defect
    
    ###
    def convexity_defect_area_e(self):
        # calculate convexity defect area / section area of the section
        defect_e = (self.pol.convex_hull.area / self.pol.area)-1
        return defect_e
    ###

    def depth_by_intersection(self):
        # calculate vertical groove depth by line detection
        # return depth and groove point
        pol_xy = self.pol.exterior.xy
        pol_r = np.asarray(pol_xy).T

        pr_1 = pol_r[pol_r[:, 1] > 0]
        point = findDeepest(pr_1)

        ymax = self.pol.bounds[3]
        d = abs(ymax - point[1])
        ext = self.circle_ext()
        return point, d / ext

    def roundness(self):
        ext = self.circle_ext()
        inn = self.circle_inn()

#         roundness = (ext - inn) / ext
        roundness = inn/ext
        return roundness

    def circle_ext(self):
        # 外接円の直径
        points = self.pol.boundary.xy
        points_tuple = []
        for i in range(np.shape(points)[1]):
            points_tuple.append(tuple([points[0][i], points[1][i]]))
        circle = smallestenclosingcircle.make_circle(points_tuple)
        return circle[2]

    def circle_inn(self):
        # convex_hullの内接円の直径
        points_hull = self.pol.convex_hull.boundary.xy  # edit 21/12/22
        points_list = []
        for i in range(np.shape(points_hull)[1]):
            points_list.append([points_hull[0][i], points_hull[1][i]])
        circle = polylabel([points_list], with_distance=True)
        return circle[1]


class VertSect(shapely.geometry.polygon.Polygon):
    """
    process a vertical section (shapely.geometry.Polygon object)
    """

    def __init__(self, pol):
        self.pol = pol
        super().__init__()

    def coord_and_curvature(self, resolution=0.5, npo=2):
        """
        :param resolution: the parameter for trimesh.path.polygons.resample_boundaries, defining minimum dist of every two points in the resampled polygon. Should be set to ensure robust curvature fitting.
        :param npo: the number of points using Calculation curvature
        ex) npo=1: using 3 point
        npo=2: using 5 point
        npo=3: using 7 point
        :return: sect_coord, sect_cur
        """
        sect_coord = trimesh.path.polygons.resample_boundaries(self.pol, resolution, [100, 2000])['shell']
        # [100, 2000]: minimum/maximum number of points in the resampled polygon
        sect_coord = anticlockwise(sect_coord)
        sect_coord = init_from_miny(sect_coord)
        sect_cur = cn.calc_curvature_circle_fitting(sect_coord[:, 0], sect_coord[:, 1], npo)
        return sect_coord, sect_cur

    @staticmethod
    def horizontal_groove_by_curvature(sect_coord, sect_cur, xrange, yrange):
        # detect deepest points of horizontal groove in v_sect
        # xrange, yrange: are the list in [min,max]
        # groove points will be explored in the ranges

        # 区間設定
        idx = []
        for e in range(len(sect_cur)):
            if xrange[0] < sect_coord[e][0] < xrange[1] and yrange[0] < sect_coord[e][1] < yrange[1]:
                idx.append(e)
        if len(idx) != 0:
            cidx = sect_cur.index(min(np.array(sect_cur)[idx]))
            groove_p = sect_coord[cidx]
            return groove_p
        else:
            return [np.nan, np.nan]

    def hgp_angle(self, gp):
        o = np.array(self.pol.centroid.xy).reshape([1, 2])
        vv = np.array([o[0][0], gp[1]])
        v1 = gp - o
        v2 = vv - o
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        prod = np.inner(v1, v2)
        cos = prod / (l1 * l2)
        rad = np.arccos(cos)
        degree = np.rad2deg(rad)
        return degree[0][0]

    def hgp_convex(self, gp, cal=True):
        """
        Return the area of convex hull defects adjacent to gp.
        Multiply line(centroid - gp) 1.1 times, and search for the adjacent areas by intersecting between the line and convex hull defects

        :param gp: groove points
        :param cal: If True, franking convex defect will be searched among the defects that don't intersect wtih y = 0
        :return:
        """
        if np.any(np.isnan(gp)):
            pol1 = []
        else:
            sec_cv = self.pol.convex_hull
            sec_dif = sec_cv.difference(self.pol)

            if cal:
                xmin, ymin, xmax, ymax = self.pol.bounds

                oline = shapely.geometry.LineString([[0, ymin], [0, ymax]])
                tree_o = STRtree(sec_dif)
                pol_o = [o for o in tree_o.query(oline) if o.intersects(oline)]
                pol_search = [o for o in sec_dif if o not in pol_o]
            else:
                pol_search = sec_dif

            o = np.array(self.pol.centroid.xy).reshape([1, 2])
            v1 = gp - o
            l1 = np.linalg.norm(v1)
            p11 = (gp * (l1 + l1 * 0.1) - o * l1 * 0.1) / l1
            v11 = shapely.geometry.LineString([gp, p11[0]])
            tree = STRtree(pol_search)
            pol1 = [o for o in tree.query(v11) if o.intersects(v11)]
        cvda = 0
        if len(pol1) > 0:
            cvda += pol1[np.argmax([o.area for o in pol1])].area
        return cvda

    def hgp_height(self, gp):
        ymax = self.pol.bounds[3]
        height = self.pol.bounds[3] - self.pol.bounds[1]
        h = (ymax - gp[1]) / height
        return h


class Sections:
    """
    process sections (list of trimesh.path2D)
    """

    def __init__(self, sections):
        self.sections = sections

    @staticmethod
    def is_null(sect):
        if sect is None:
            return True
        elif sect.body_count == 0:  # edit 211213
            return True
        else:
            return False

    @staticmethod
    def select_largest_polygon(sect):
        # return shapely.geometry.Polygon objects
        area = []
        for pol in sect.polygons_closed:
            try:
                area.append(pol.area)
            except AttributeError:
                area.append(0)
        pol_id = np.argmax(area)
        respol = sect.polygons_closed[pol_id]
        return respol

    def convexity_defects_area(self):
        res_convex_defect = []
        for i in self.sections:
            if not self.is_null(i):
                pol = TransSect(self.select_largest_polygon(i))
                res_convex_defect.append(pol.convexity_defect_area())
            else:
                res_convex_defect.append(np.nan)
        return res_convex_defect
    
        
    def convexity_defects_area_e(self):
        res_convex_defect_e = []
        for i in self.sections:
            if not self.is_null(i):
                pol = TransSect(self.select_largest_polygon(i))
                res_convex_defect_e.append(pol.convexity_defect_area_e())
            else:
                res_convex_defect_e.append(np.nan)
        return res_convex_defect_e

    def transverse_area(self):
        res_area = []
        curr = 0
        for i in self.sections:
            if not self.is_null(i):
                res_area.append(i.area)
            else:
                res_area.append(np.nan)
            curr += 1
        return res_area

    def roundness(self):
        res_area = []
        for i in self.sections:
            if not self.is_null(i):
                pol = TransSect(self.select_largest_polygon(i))
                res_area.append(pol.roundness())
            else:
                res_area.append(np.nan)
        return res_area

    def trans_coordinate_direction(self, t_conv):
        # max_id = res_area.index(max(res_area))
        # if max_id > 50:
        #    sections.reverse()
        #    res_area = transverse_area(sections)
        # return sections, res_area

        # t_conv=1のとき、sectionsを逆にする
        if t_conv == 1:
            self.sections.reverse()

    def vert_coordinate_direction(self):
        sections_oriented = []
        # curr = 0
        d_sects = copy.deepcopy(self.sections)
        rotates = []

        for sect in d_sects:
            if not self.is_null(sect):
                sect.fill_gaps()
                pol = self.select_largest_polygon(sect)
                pol_cv = pol.convex_hull
                pol_dif = pol_cv.difference(pol)

                calyx_idx = np.argmax([r.area for r in pol_dif])
                x, y = pol_dif[calyx_idx].centroid.xy

                if x > y:
                    if abs(x[0]) > abs(y[0]):  # へたが右
                        arr = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                    else:  # へたが下
                        arr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
                else:
                    if abs(x[0]) > abs(y[0]):  # へたが左
                        arr = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                    else:
                        arr = np.identity(3)
                sect.apply_transform(arr)
                rotates.append(arr)
                # へたが上はそのまま
                # curr += 1
            sections_oriented.append(sect)
        return sections_oriented, rotates

    def depth_by_intersection_vis(self, step):

        nrow = 2  # can be changed
        num_sect = len(self.sections)
        if step != 0:
            ncol = int(num_sect / step / nrow)
            fig = plt.figure(1, figsize=(10, 5))
            fig.tight_layout()

        polygon_sample_resolution = 5  # 軽くするため、数を減らす。trimesh.path.polygons.resample_boundaries()の説明をみる

        curr = 0
        res_depth = []
        d_sects = copy.deepcopy(self.sections)
        for i in d_sects:
            if self.is_null(i):
                res_depth.append(np.nan)
            elif not i.is_closed:
                res_depth.append(np.nan)
            else:
                i.fill_gaps()
                i.apply_obb()
                [[xmin, ymin], [xmax, ymax]] = i.bounds
                points = []
                depth = []
                sect_ori = copy.deepcopy(i)
                for x in range(4):
                    # 最大ポリゴンを見つける
                    i = rotate90(i)  # 回転
                    pol = TransSect(self.select_largest_polygon(i))
                    point, d = pol.depth_by_intersection()
                    depth.append(d)
                    for y in range(3 - x):
                        point = rotate_origin(point, np.pi / 2)  # 90°回転、何度か回して、元の座標に戻す
                    points.append(point)
                res_depth.append(np.average(depth))
                i = sect_ori

                if not step == 0 and curr % step == 0:
                    ax = fig.add_subplot(nrow, ncol, curr // step + 1)
                    ax.set_aspect('equal')
                    ax.set_ylim(ymin, ymax)
                    ax.set_xlim(xmin, xmax)
                    ax.set_title("section " + str(curr), fontsize=8)
                    ax.tick_params(labelsize=8)

                    ax.set_ylim(ymin, ymax)
                    ax.set_xlim(xmin, xmax)
                    if not self.is_null(i):
                        pol = self.select_largest_polygon(i)
                        patch = patches.Polygon(pol.exterior, alpha=0.3)
                        ax.add_patch(patch)
                    for z in range(4):
                        ax.plot(points[z][0], points[z][1], marker="o", markersize=2, color="red")
                        ax.text(points[z][0], points[z][1], str(round(depth[z], 2)), fontsize=6)
            curr += 1
        return res_depth

    def show_combined_sections(self):
        combined = np.sum(self.sections)
        plt.figure(2)
        combined.show()

    def vert_get_lw(self):
        res_lw = []
        for sect in self.sections:
            if self.is_null(sect):
                res_lw.append(np.nan)
            else:
                [[xmin, ymin], [xmax, ymax]] = sect.bounds
                res_lw.append((ymax - ymin) / (xmax - xmin))
        return res_lw

    def v_horizontal_groove(self, init_limit=0.3, resolution=0.5, npo=2, cda_thre=0.0001, vis=True):
        """
        param vis: boolean True to turn on visualization, False to turn it off.

        :param init_limit: variable that specifies region not to search for the groove point in initial sampling. This doesn't search between 0 and xmax*init_limit in x.
        :param resolution: Minimum distance for contour point sampling
        :param npo: How many points to approximate when computing curvature. 1=3 points, 2=5 points, 3=7 points
        :param cda_thre: convexity defect area threshold
        :param vis: visualization
        :return:
        """
        tgroove_p1_tmp = []
        tgroove_p2_tmp = []
        num_sect = len(self.sections)
        err1, err2 = 0, 0

        count_gp = 0

        for sect in self.sections:
            if self.is_null(sect):
                continue
            large_pol = self.select_largest_polygon(sect)  # modified 220608
            pol = VertSect(large_pol)
            sect_coord, sect_cur = pol.coord_and_curvature(resolution, npo)
            xmin, ymin = np.min(sect_coord, axis=0)
            xmax, ymax = np.max(sect_coord, axis=0)

            p1 = VertSect.horizontal_groove_by_curvature(sect_coord, sect_cur, [xmax * init_limit, xmax], [0, ymax])
            cvda1 = pol.hgp_convex(p1)
            p2 = VertSect.horizontal_groove_by_curvature(sect_coord, sect_cur, [xmin, xmin * init_limit], [0, ymax])
            cvda2 = pol.hgp_convex(p2)
            if cvda1 / large_pol.area > cda_thre:
                tgroove_p1_tmp.append(p1)
                count_gp += 1
            else:
                err1 += 1
            if cvda2 / large_pol.area > cda_thre:
                tgroove_p2_tmp.append(p2)
                count_gp += 1
            else:
                err2 += 1
        print("1st round gp detection: " + str(count_gp) + " / " + str(num_sect * 2) + " points")

        if err1 / num_sect > 0.8 or err2 / num_sect > 0.8:
            print("total vert sections: " + str(num_sect) + "\nfailed potential gp detection right: " + str(
                err1) + ", left: " + str(err2))
            tg_angle = np.zeros(num_sect * 2)
            tg_cvd = np.zeros(num_sect * 2)
            tg_h = np.zeros(num_sect * 2)
            tgroove_p1 = 0
            tgroove_p2 = 0

        else:
            # 一回目探索の座点の座標から、二回目探索の範囲を決定
            label1 = self.tg_point_gmeans(np.array(tgroove_p1_tmp))
            mode1 = self.mode_label(label1)
            xmin1, ymin1, xmax1, ymax1 = self.second_search_range(np.array(tgroove_p1_tmp), mode1, 0.1)
            # xmin1, ymin1, xmax1, ymax1 = self.second_search_range(np.array(tgroove_p1_tmp), range(len(tgroove_p1_tmp)), 0.1)

            label2 = self.tg_point_gmeans(np.array(tgroove_p2_tmp))
            mode2 = self.mode_label(label2)
            xmin2, ymin2, xmax2, ymax2 = self.second_search_range(np.array(tgroove_p2_tmp), mode2, 0.1)
            # xmin2, ymin2, xmax2, ymax2 = self.second_search_range(np.array(tgroove_p2_tmp), range(len(tgroove_p2_tmp)), 0.1)

            # 二回目探索
            tgroove_p1 = []
            tgroove_p2 = []
            tg_angle = []
            tg_cvd = []
            tg_h = []

            fig = plt.figure(3, figsize=(10, 10))
            step = 1
            horizontal_count = 5
            vertical_count = int(len(self.sections) / step / horizontal_count) + 1
            curr = 0

            count_gp = 0

            for sect in self.sections:
                curr += 1
                if self.is_null(sect):
                    continue
                pol = VertSect(self.select_largest_polygon(sect))

                [[xmin, ymin], [xmax, ymax]] = sect.bounds
                ax = fig.add_subplot(vertical_count, horizontal_count, curr)
                ax.set_aspect('equal')
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(xmin, xmax)
                ax.set_title("section " + str(curr - 1), fontsize=5)
                ax.tick_params(labelsize=5)
                # sect_polygon = sect.polygons_full
                large_pol = self.select_largest_polygon(sect)
                # if len(sect_polygon) == 0:
                #    continue
                # for j in range(len(sect_polygon)):
                #    patch = patches.Polygon(sect_polygon[j].exterior.coords, fc='gray', ec='k', alpha=0.2)
                #    ax.add_patch(patch)

                patch = patches.Polygon(large_pol.exterior.coords, fc='gray', ec='k', alpha=0.2)
                ax.add_patch(patch)

                sect_coord, sect_cur = pol.coord_and_curvature(resolution, npo)

                p1 = VertSect.horizontal_groove_by_curvature(sect_coord, sect_cur, [xmin1, xmax1], [ymin1, ymax1])
                cvda1 = pol.hgp_convex(p1, True)
                p2 = VertSect.horizontal_groove_by_curvature(sect_coord, sect_cur, [xmin2, xmax2], [ymin2, ymax2])
                cvda2 = pol.hgp_convex(p2, True)
                if cvda1 / large_pol.area > cda_thre:
                    ax.scatter(p1[0], p1[1])
                    count_gp += 1
                else:
                    p1 = [np.nan, np.nan]
                if cvda2 / large_pol.area > cda_thre:
                    ax.scatter(p2[0], p2[1])
                    count_gp += 1
                else:
                    p2 = [np.nan, np.nan]

                tgroove_p1.append(p1)
                tgroove_p2.append(p2)

                pts = [p1, p2]

                for pt in pts:
                    if not np.any(np.isnan(pt)):
                        angle = pol.hgp_angle(pt)
                        cvd = pol.hgp_convex(pt)
                        height = pol.hgp_height(pt)
                    else:
                        angle = np.nan
                        cvd = np.nan
                        height = np.nan
                    tg_angle.append(angle)
                    tg_cvd.append(cvd)
                    tg_h.append(height)

            fig.tight_layout()
            if not vis:
                plt.close()
            print("2nd round gp detection: " + str(count_gp) + " / " + str(num_sect * 2) + " points")
        return tg_angle, tg_cvd, tg_h, tgroove_p1, tgroove_p2

    @staticmethod
    def tg_point_gmeans(tg_points):
        # 座点候補のgmeans
        gmeans_instance = gmeans.gmeans(tg_points).process()

        clusters = gmeans_instance.get_clusters()

        labels_size = len(
            list(itertools.chain.from_iterable(clusters))
        )

        labels = np.zeros((1, labels_size))
        if len(clusters) != 1:
            for n, n_th_cluster in np.ndenumerate(clusters):
                for img_num in n_th_cluster:
                    labels[0][img_num] = n[0]
        labels = labels.ravel()
        return labels

    @staticmethod
    def mode_label(labels):
        # gmeans結果の最頻値
        unique, freq = np.unique(labels, return_counts=True)
        mode = unique[np.argmax(freq)]
        ref_idx = [i for i in range(len(labels)) if labels[i] == mode]
        return ref_idx

    @staticmethod
    def second_search_range(tg_points, mode_idx, k=0.1):
        xmin, ymin = np.min(tg_points[mode_idx], axis=0)
        xmax, ymax = np.max(tg_points[mode_idx], axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        xmin = xmin - dx * k
        ymin = ymin - dy * k
        xmax = xmax + dx * k
        ymax = ymax + dy * k
        return xmin, ymin, xmax, ymax
