"""
main functions for fruit 3D measurement of persimmon

"""
import proc2d as pr2
from descartes.patch import PolygonPatch
from matplotlib import pyplot as plt


def show_center_sliced_section(mesh):
    fig = plt.figure(5,figsize=(5, 10))
    max_retry = 5
    lw = []

    for i in range(3):
        plane = [0, 0, 0]
        plane[i] = 1
        for c in range(max_retry):
            try:
                slice1 = mesh.section(plane_origin=mesh.centroid + c, plane_normal=plane)
                slice2, to_3D = slice1.to_planar()
                if slice2.is_closed:
                    break
            except IndexError:
                print("section #" + str(i) + " retry " + str(c))
                pass
            except AttributeError:
                print("section #" + str(i) + " retry " + str(c))
                pass
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_aspect('equal')
        [[xmin, ymin], [xmax, ymax]] = slice2.bounds
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        patch1 = PolygonPatch(slice2.polygons_full[0])
        ax.add_patch(patch1)

        d = pr2.get_lw(slice2)
        lw.append(abs(d - 1))

    fig.tight_layout()
    #fig.show()

    return mesh, lw


