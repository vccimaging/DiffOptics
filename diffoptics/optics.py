from .basics import *
from .shapes import *
from scipy.interpolate import LSQBivariateSpline
import matplotlib.pyplot as plt
import copy
import pathlib


def tex(img_2d, size_2d, x, y, bmode=BoundaryMode.replicate): # texture indexing function
    if bmode is BoundaryMode.zero:
        raise NotImplementedError()
    elif bmode is BoundaryMode.replicate:
        x = torch.clamp(x, min=0, max=size_2d[0]-1)
        y = torch.clamp(y, min=0, max=size_2d[1]-1)
    elif bmode is BoundaryMode.symmetric:
        raise NotImplementedError()
    elif bmode is BoundaryMode.periodic:
        raise NotImplementedError()
    img = img_2d[x.flatten(), y.flatten()]
    return img.reshape(x.shape)

def tex4(img_2d, size_2d, x0, y0, bmode=BoundaryMode.replicate): # texture indexing four pixels
    _tex = lambda x, y : tex(img_2d, size_2d, x, y, bmode)
    s00 = _tex(  x0,   y0)
    s01 = _tex(  x0, 1+y0)
    s10 = _tex(1+x0,   y0)
    s11 = _tex(1+x0, 1+y0)
    return s00, s01, s10, s11


class Lensgroup(Endpoint):
    """
    The origin of the Lensgroup, which is a collection of multiple optical surfaces, is located at "origin".
    The Lensgroup can rotate freely around the x/y axes, and the rotation angles are defined as "theta_x", "theta_y", and "theta_z" (in degrees).
    
    In the Lensgroup's coordinate system, which is the object frame coordinate system, surfaces are arranged starting from "z = 0".
    There is a small 3D origin shift, called "shift", between the center of the surface (0,0,0) and the mount's origin.
    The sum of the shift and the origin is equal to the Lensgroup's origin.
    
    There are two configurations for ray tracing: forward and backward.
    - In the forward mode, rays begin at the surface with "d = 0" and propagate along the +z axis, e.g. from scene to image plane.
    - In the backward mode, rays begin at the surface with "d = d_max" and propagate along the -z axis, e.g. from image plane to scene.
    """
    def __init__(self, origin=np.zeros(3), shift=np.zeros(3), theta_x=0., theta_y=0., theta_z=0., device=torch.device('cpu')):
        self.origin = torch.Tensor(origin).to(device)
        self.shift = torch.Tensor(shift).to(device)
        self.theta_x = torch.Tensor(np.asarray(theta_x)).to(device)
        self.theta_y = torch.Tensor(np.asarray(theta_y)).to(device)
        self.theta_z = torch.Tensor(np.asarray(theta_z)).to(device)
        self.device = device

        # Sequentials properties
        self.surfaces = []
        self.materials = []
        
        # Sensor properties
        self.pixel_size = 6.45 # [um]
        self.film_size = [640, 480] # [pixel]

        Endpoint.__init__(self, self._compute_transformation(), device)

        # TODO: in case you would like to render something in Mitsuba2 ...
        self.mts_prepared = False

    def load_file(self, filename: pathlib.Path):
        self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(str(filename))
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def load(self, surfaces: list, materials: list):
        self.surfaces = surfaces
        self.materials = materials
        self._sync()

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)
        self.aperture_ind = self._find_aperture()

    def update(self, _x=0.0, _y=0.0):
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()
    
    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        # we compute to_world transformation given the input positional parameters (angles)
        R = ( rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x+_x)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y+_y)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z+_z)) )
        t = self.origin + R @ self.shift
        return Transformation(R, t)
    
    def _find_aperture(self):
        for i in range(len(self.surfaces)-1):
            if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # both are AIR
                return i

    @staticmethod
    def read_lensfile(filename):
        surfaces = []
        materials = []
        ds = [] # no use for now
        with open(filename) as file:
            line_no = 0
            d_total = 0.
            for line in file:
                if line_no < 2: # first two lines are comments; ignore them
                    line_no += 1 
                else:
                    ls = line.split()
                    surface_type, d, r = ls[0], float(ls[1]), float(ls[3])/2
                    roc = float(ls[2])
                    if roc != 0: roc = 1/roc
                    materials.append(Material(ls[4]))
                    
                    d_total += d
                    ds.append(d)

                    if surface_type == 'O': # object
                        d_total = 0.
                        ds.pop()
                    elif surface_type == 'X': # XY-polynomial
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                b = float(ls[5])
                            else:
                                ai.append(float(ls[ac]))
                        surfaces.append(XYPolynomial(r, d_total, J=3, ai=ai, b=b))
                    elif surface_type == 'B': # B-spline
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                nx = int(ls[5])
                            elif ac == 6:
                                ny = int(ls[6])
                            else:
                                ai.append(float(ls[ac]))
                        tx = ai[:nx+8]
                        ai = ai[nx+8:]
                        ty = ai[:ny+8]
                        ai = ai[ny+8:]
                        c  = ai
                        surfaces.append(BSpline(r, d, size=[nx, ny], tx=tx, ty=ty, c=c))
                    elif surface_type == 'M': # mixed-type of X and B
                        raise NotImplementedError()
                    elif surface_type == 'S': # aspheric surface
                        if len(ls) <= 5:
                            surfaces.append(Aspheric(r, d_total, roc))
                        else:
                            ai = []
                            for ac in range(5, len(ls)):
                                if ac == 5:
                                    conic = float(ls[5])
                                else:
                                    ai.append(float(ls[ac]))
                            surfaces.append(Aspheric(r, d_total, roc, conic, ai))
                    elif surface_type == 'A': # aperture
                        surfaces.append(Aspheric(r, d_total, roc))
                    elif surface_type == 'I': # sensor
                        d_total -= d
                        ds.pop()
                        materials.pop()
                        r_last = r
                        d_last = d
        return surfaces, materials, r_last, d_last

    def reverse(self):
        # reverse surfaces
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()
        self.surfaces.reverse()
        
        # reverse materials
        self.materials.reverse()

    # ------------------------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------------------------
    def rms(self, ps, units=1, option='centroid', squared=False):
        ps = ps[...,:2] * units
        if option == 'centroid':
            ps_mean = torch.mean(ps, axis=0)
        ps = ps - ps_mean[None,...] # we now use normalized ps
        if squared:
            return torch.mean(torch.sum(ps**2, axis=-1)), ps/units
        else:
            return torch.sqrt(torch.mean(torch.sum(ps**2, axis=-1))), ps/units

    def spot_diagram(self, ps, show=True, xlims=None, ylims=None, color='b.', savepath=None):
        """
        Plot spot diagram.
        """
        units = 1
        spot_rms = float(self.rms(ps, units)[0])
        ps = ps.cpu().detach().numpy()[...,:2]
        ps_mean = np.mean(ps, axis=0) # centroid
        ps = ps - ps_mean[None,...] # we now use normalized ps
        
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ps[...,1], ps[...,0], color)
        plt.gca().set_aspect('equal', adjustable='box')

        if xlims is not None:
            plt.xlim(*xlims)
        if ylims is not None:
            plt.ylim(*ylims)
        ax.set_aspect(1./ax.get_data_ratio())
        units_str = '[mm]'
        plt.xlabel('x ' + units_str)
        plt.ylabel('y ' + units_str)
        plt.xticks(np.linspace(xlims[0], xlims[1], 11))
        plt.yticks(np.linspace(ylims[0], ylims[1], 11))
        # plt.grid(True)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()
        
        return spot_rms
    
    def split(self, combine_flags=None):
        """
        Split a lensgroup into several smaller lensgroups.
        """
        def create_lens(indices):
            d = float(self.surfaces[indices[0]].d.cpu().detach().numpy())
            lens = Lensgroup(
                origin=np.array([0.0,0.0,d]), #_compute_transformation
                shift=np.zeros(3), 
                theta_x=0.0,
                theta_y=0.0,
                theta_z=0.0,
                device=self.device
            )
            lens.load( # here we make a deep copy to make a copy for every surface & material
                copy.deepcopy([self.surfaces[j] for j in indices]),
                copy.deepcopy([self.materials[j] for j in indices + [indices[-1]+1]])
            )
            # de-center the surfaces
            for i in range(len(lens.surfaces)):
                lens.surfaces[i].d = lens.surfaces[i].d - d
            return lens
        
        lenses = []
        indices = []
        k = 0
        for i in range(len(self.surfaces)):
            if self.materials[i].A < 1.0003: # AIR
                if i > 0: # ends of a lensgroup
                    if combine_flags is not None:
                        k += 1
                        if combine_flags[k]: # skip this one
                            indices.append(i)
                            continue
                        else:
                            lenses.append(create_lens(indices))
                            indices = []
                    else:
                        lenses.append(create_lens(indices))
                        indices = []
                indices.append(i)
            else:
                indices.append(i)

        # end of a lensgroup
        lenses.append(create_lens(indices))
        
        return lenses
    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    # IO and visualizations
    # ------------------------------------------------------------------------------------
    def draw_points(self, ax, options, seq=range(3)):
        for surface in self.surfaces:
            points_world = self._generate_points(surface)
            ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)

    def get_lines_from_plot_setup2D(self, with_sensor=True):
        lines = []

        # to world coordinate
        def plot(lines: list, surface_id, z, x):
            p = self.to_world.transform_point(
                torch.stack(
                    (x, torch.zeros_like(x, device=self.device), z), axis=-1
                )
            ).cpu().detach().numpy()
            lines.append({'z': p[...,2], 'x': p[...,0], 'id': surface_id})

        def draw_aperture(lines: list, surface, surface_id):
            N = 3
            d = surface.d.cpu()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(lines, surface_id, z, x)
            x = R * torch.ones(N, device=self.device)
            plot(lines, surface_id, z, x)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(lines, surface_id, z, x)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(lines, surface_id, z, x)

        if len(self.surfaces) == 1: # if there is only one surface, then it has to be the aperture
            draw_aperture(lines, self.surfaces[0], 0)
        else:
            # draw sensor plane
            if with_sensor == True:
                try:
                    tmpr, tmpdd = self.r_last, self.d_sensor
                except AttributeError:
                    with_sensor = False

            if with_sensor:
                self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))

            # draw surface
            for i, s in enumerate(self.surfaces):
                # find aperture
                if i < len(self.surfaces)-1:
                    if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # both are AIR
                        draw_aperture(lines, s, i)
                        continue
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))
                plot(lines, i, z, r)
            
            # draw boundary
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003: # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([r_prev, r])).to(self.device)
                    plot(lines, i, z, x)
                    plot(lines, i, z,-x)
                    s_prev = s
            
            # remove sensor plane
            if with_sensor:
                self.surfaces.pop()

        return lines

    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True):
        """
        Plot elements in 2D.
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            show=False

        # to world coordinate
        def plot(ax, z, x, color):
            p = self.to_world.transform_point(
                torch.stack(
                    (x, torch.zeros_like(x, device=self.device), z), axis=-1
                )
            ).cpu().detach().numpy()
            ax.plot(p[...,2], p[...,0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d.cpu()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)

        if len(self.surfaces) == 1: # if there is only one surface, then it has to be the aperture
            draw_aperture(ax, self.surfaces[0], color)
        else:
            # draw sensor plane
            if with_sensor:
                try:
                    self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))
                except AttributeError:
                    with_sensor = False

            # draw surface
            for i, s in enumerate(self.surfaces):
                # find aperture
                if i < len(self.surfaces)-1:
                    if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # both are AIR
                        draw_aperture(ax, s, color)
                        continue
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))
                plot(ax, z, r, color)
            
            # draw boundary
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003: # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([r_prev, r])).to(self.device)
                    plot(ax, z, x, color)
                    plot(ax, z,-x, color)
                    s_prev = s
            
            # remove sensor plane
            if with_sensor:
                self.surfaces.pop()

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        plt.title("Layout 2D")
        if show: plt.show()
        return ax, fig

    # TODO: modify the tracing part to include oss
    def plot_raytraces(self, oss, ax=None, fig=None, color='b-', show=True, p=None, valid_p=None):
        """
        Plot all ray traces (oss).
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D(show=False)
        else:
            show=False
        for i, os in enumerate(oss):
            o = torch.Tensor(np.array(os)).to(self.device)
            x = o[...,0]
            z = o[...,2]

            # to world coordinate
            o = self.to_world.transform_point(
                torch.stack(
                    (x, torch.zeros_like(x, device=self.device), z), axis=-1
                )
            )
            o = o.cpu().detach().numpy()
            z = o[...,2].flatten()
            x = o[...,0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i,0])
                    z = np.append(z, p[i,2])

            ax.plot(z, x, color, linewidth=1.0)

        if show: plt.show()
        else: plt.close()
        return ax, fig

    def plot_setup2D_with_trace(self, views, wavelength, M=2, R=None, entrance_pupil=True):
        if R is None:
            R = self.surfaces[0].r
        colors_list = 'bgrymck'
        ax, fig = self.plot_setup2D(show=False)

        for i, view in enumerate(views):
            ray = self.sample_ray_2D(R, wavelength, view=view, M=M, entrance_pupil=entrance_pupil)
            ps, oss = self.trace_to_sensor_r(ray)
            ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i])

        # fig.show()
        return ax, fig

    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------------------
    def calc_entrance_pupil(self, view=0.0, R=None):
        angle = np.radians(np.asarray(view))
        
        # maximum radius input
        if R is None:
            with torch.no_grad():
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0)
                R = np.tan(angle) * sag + self.surfaces[0].r # [mm]
                R = R.item()

        APERTURE_SAMPLING = 101
        x, y = torch.meshgrid(
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )

        # generate rays and find valid map
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        o = torch.stack((x,y,zeros), axis=2)
        d = torch.stack((
            np.sin(angle)*ones,
            zeros,
            np.cos(angle)*ones), axis=-1
        )
        ray = Ray(o, d, torch.Tensor([580.0]).to(self.device), device=self.device)
        valid_map = self.trace_valid(ray)

        # find bounding box
        xs, ys = x[valid_map], y[valid_map]

        return valid_map, xs, ys

    def sample_ray(self, wavelength, view=0.0, M=15, R=None, shift_x=0., shift_y=0., sampling='grid', entrance_pupil=False):
        angle = np.radians(np.asarray(view))
        
        # maximum radius input
        if R is None:
            with torch.no_grad():
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0)
                R = np.tan(angle) * sag + self.surfaces[0].r # [mm]
                R = R.item()

        if entrance_pupil:
            xs, ys = self.calc_entrance_pupil(view, R)[1:]
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(xs.min(), xs.max(), M, device=self.device),
                    torch.linspace(ys.min(), ys.max(), M, device=self.device),
                    indexing='ij'
                )
            elif sampling == 'radial':
                R = np.minimum(xs.max() - xs.min(), ys.max() - ys.min())
                r = torch.linspace(0, R, M, device=self.device)
                theta = torch.linspace(0, 2*np.pi, M+1, device=self.device)[0:M]
                x = xs.mean() + r[None,...] * torch.cos(theta[...,None])
                y = ys.mean() + r[None,...] * torch.sin(theta[...,None])
        else:
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M, device=self.device),
                    torch.linspace(-R, R, M, device=self.device),
                    indexing='ij'
                )
            elif sampling == 'radial':
                r = torch.linspace(0, R, M, device=self.device)
                theta = torch.linspace(0, 2*np.pi, M+1, device=self.device)[0:M]
                x = r[None,...] * torch.cos(theta[...,None])
                y = r[None,...] * torch.sin(theta[...,None])

        p = 2*R / M
        x = x + p * shift_x
        y = y + p * shift_y

        o = torch.stack((x,y,torch.zeros_like(x, device=self.device)), axis=2)
        d = torch.stack((
            np.sin(angle)*torch.ones_like(x),
            torch.zeros_like(x),
            np.cos(angle)*torch.ones_like(x)), axis=-1
        )
        return Ray(o, d, wavelength, device=self.device)
    
    # TODO: merge `sample_ray_fullfield` with `sample_ray`
    def sample_ray_fullfield(self, wavelength, view_xy=[0.0,0.0], M=15, R=None, shift_xy=[0.,0.], sampling='grid'):
        angle_xy = torch.Tensor(np.radians(np.asarray(view_xy))).to(self.device)
        if sampling == 'grid':
            x, y = torch.meshgrid(
                torch.linspace(-R, R, M, device=self.device),
                torch.linspace(-R, R, M, device=self.device),
                indexing='ij'
            )
        elif sampling == 'radial':
            r = torch.linspace(0, R, M, device=self.device)
            theta = torch.linspace(0, 2*np.pi, M+1, device=self.device)[0:M]
            x = r[None,...] * torch.cos(theta[...,None])
            y = r[None,...] * torch.sin(theta[...,None])

        p = 2*R / M
        x = x + p * shift_xy[0]
        y = y + p * shift_xy[1]

        o = torch.stack((x,y,torch.zeros_like(x, device=self.device)), axis=2)
        d = torch.stack((
            torch.sin(angle_xy[0])*torch.ones_like(x),
            torch.sin(angle_xy[1])*torch.ones_like(x),
            torch.cos(angle_xy[0])*torch.cos(angle_xy[1])*torch.ones_like(x)), axis=-1
        )
        return Ray(o, d, wavelength, device=self.device)

    def sample_ray_2D(self, R, wavelength, view=0.0, M=15, shift_x=0., entrance_pupil=False):
        if entrance_pupil:
            # x_up, x_down, x_center = self.find_ray_2D(view=view)
            xs = self.calc_entrance_pupil(view=view)[1]
            x_up = xs.min()
            x_down = xs.max()
            x_center = xs.mean()

            x = torch.hstack((
                torch.linspace(x_down, x_center, M+1, device=self.device)[:M],
                torch.linspace(x_center, x_up, M+1, device=self.device),
            ))
        else:
            x = torch.linspace(-R, R, M, device=self.device)
        p = 2*R / M
        x = x + p * shift_x

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        
        o = torch.stack((x,zeros,zeros), axis=1)
        angle = torch.Tensor(np.asarray(np.radians(view))).to(self.device)
        d = torch.stack((
            torch.sin(angle)*ones,
            zeros,
            torch.cos(angle)*ones), axis=-1
        )
        return Ray(o, d, wavelength, device=self.device)
    
    def find_ray_2D(self, view=0.0, y=0.0):
        """
        This function finds chief and marginal rays at a specific view.
        """
        wavelength = torch.Tensor([589.3]).to(self.device)
        R_aperture = self.surfaces[self.aperture_ind].r
        angle = np.radians(view)
        d = torch.Tensor(np.stack((
            np.sin(angle),
            y,
            np.cos(angle)), axis=-1
        )).to(self.device)

        def find_x(alpha=1.0): # TODO: does not work for wide-angle lenses!
            x = - np.tan(angle) * self.surfaces[self.aperture_ind].d.cpu().detach().numpy()
            is_converge = False
            for k in range(30):
                o = torch.Tensor([x, y, 0.0])
                ray = Ray(o, d, wavelength, device=self.device)
                ray_final, valid = self.trace(ray, stop_ind=self.aperture_ind)[:2]
                x_aperture = ray_final.o[0].cpu().detach().numpy()
                diff = 0.0 - x_aperture
                if np.abs(diff) < 0.001:
                    print('`find_x` converges!')
                    is_converge = True
                    break
                if valid:
                    x_last = x
                    if diff > 0.0:
                        x += alpha * diff
                    else:
                        x -= alpha * diff
                else:
                    x = (x + x_last)/2
            return x, is_converge
        
        def find_bx(x_center, R_aperture, alpha=1.0):
            x = x_center
            x_last = 0.0 # temp
            for k in range(100):
                o = torch.Tensor([x, y, 0.0])
                ray = Ray(o, d, wavelength, device=self.device)
                ray_final, valid = self.trace(ray, stop_ind=self.aperture_ind)[:2]
                x_aperture = ray_final.o[0].cpu().detach().numpy()
                diff = R_aperture - x_aperture
                if np.abs(diff) < 0.01:
                    print('`find_x` converges!')
                    break
                if valid:
                    x_last = x
                    if diff > 0.0:
                        x += alpha * diff
                    else:
                        x -= alpha * diff
                else:
                    x = (x + x_last)/2
            return x_last
        
        x_center, is_converge = find_x(alpha=-np.sign(view)*1.0)
        if not is_converge:
            x_center, is_converge = find_x(alpha=np.sign(view)*1.0)

        x_up = find_bx(x_center, R_aperture, alpha=1)
        x_down = find_bx(x_center, -R_aperture, alpha=-1)
        return x_up, x_down, x_center

    # ------------------------------------------------------------------------------------
    
    def render(self, ray, irr=1.0):
        """
        Forward rendering.
        """
        # TODO: remind users to prepare filmsize and pixelsize before using this function.
                    
        # trace rays
        ray_final, valid = self.trace(ray)

        # intersecting sensor plane
        t = (self.d_sensor - ray_final.o[...,2]) / ray_final.d[...,2]
        p = ray_final(t)

        R_sensor = [self.film_size[i] * self.pixel_size/2 for i in range(2)]
        valid = valid & (
            (-R_sensor[0] <= p[...,0]) & (p[...,0] <= R_sensor[0]) &
            (-R_sensor[1] <= p[...,1]) & (p[...,1] <= R_sensor[1])
        )

        # intensity
        J = irr
        p = p[valid]

        # compute shift and find nearest pixel index
        u = (p[...,0] + R_sensor[0]) / self.pixel_size
        v = (p[...,1] + R_sensor[1]) / self.pixel_size
        
        index_l = torch.stack(
            (torch.clamp(torch.floor(u).long(), min=0, max=self.film_size[0]-1),
            torch.clamp(torch.floor(v).long(), min=0, max=self.film_size[1]-1)),
            axis=-1
        )
        index_r = torch.stack(
            (torch.clamp(index_l[...,0] + 1, min=0, max=self.film_size[0]-1),
            torch.clamp(index_l[...,1] + 1, min=0, max=self.film_size[1]-1)),
            axis=-1
        )
        w_r = torch.clamp(torch.stack((u,v), axis=-1) - index_l, min=0, max=1)
        w_l = 1.0 - w_r
        del u, v

        # compute image
        I = torch.zeros(*self.film_size, device=self.device)
        I = torch.index_put(I, (index_l[...,0],index_l[...,1]), w_l[...,0]*w_l[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_l[...,1]), w_r[...,0]*w_l[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_l[...,0],index_r[...,1]), w_l[...,0]*w_r[...,1]*J, accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_r[...,1]), w_r[...,0]*w_r[...,1]*J, accumulate=True)
        return I

    def trace_valid(self, ray):
        """
        Trace rays to see if they intersect the sensor plane or not.
        """
        valid = self.trace(ray)[1]
        return valid

    def trace_to_sensor(self, ray, ignore_invalid=False):
        """
        Trace rays towards intersecting onto the sensor plane.
        """
        # trace rays
        ray_final, valid = self.trace(ray)

        # intersecting sensor plane
        t = (self.d_sensor - ray_final.o[...,2]) / ray_final.d[...,2]
        p = ray_final(t)
        if ignore_invalid:
            p = p[valid]
        else:
            if len(p.shape) < 2:
                return p
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))
        return p
        
    def trace_to_sensor_r(self, ray, ignore_invalid=False):
        """
        Trace rays towards intersecting onto the sensor plane, with records.
        """
        # trace rays
        ray_final, valid, oss = self.trace_r(ray)

        # intersecting sensor plane
        t = (self.d_sensor - ray_final.o[...,2]) / ray_final.d[...,2]
        p = ray_final(t)
        if ignore_invalid:
            p = p[valid]
        else:
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))
        
        for v, os, pp in zip(valid, oss, p):
            if v:
                os.append(pp.cpu().detach().numpy())

        return p, oss

    def trace(self, ray, stop_ind=None):
        # update transformation when doing pose estimation
        if (
            self.origin.requires_grad
            or
            self.shift.requires_grad
            or
            self.theta_x.requires_grad
            or
            self.theta_y.requires_grad
            or
            self.theta_z.requires_grad
        ):
            self.update()

        # in local
        ray_in = self.to_object.transform_ray(ray)

        valid, ray_out = self._trace(ray_in, stop_ind=stop_ind, record=False)
        
        # in world
        ray_final = self.to_world.transform_ray(ray_out)

        return ray_final, valid

    def trace_r(self, ray, stop_ind=None):
        # update transformation when doing pose estimation
        if (
            self.origin.requires_grad
            or
            self.shift.requires_grad
            or
            self.theta_x.requires_grad
            or
            self.theta_y.requires_grad
            or
            self.theta_z.requires_grad
        ):
            self.update()

        # in local
        ray_in = self.to_object.transform_ray(ray)

        valid, ray_out, oss = self._trace(ray_in, stop_ind=stop_ind, record=True)
        
        # in world
        ray_final = self.to_world.transform_ray(ray_out)
        for os in oss:
            for o in os:
                os = self.to_world.transform_point(torch.Tensor(np.asarray(os)).to(self.device)).cpu().detach().numpy()

        return ray_final, valid, oss

    # ------------------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------------------
    def prepare_mts(self, pixel_size, film_size, R=np.eye(3), t=np.zeros(3)):
        # TODO: this is actually prepare_backward tracing ...
        """
        Revert surfaces for Mitsuba2 rendering.
        """
        if self.mts_prepared:
            print('MTS already prepared for this lensgroup.')
            return
            
        # sensor parameters
        self.pixel_size = pixel_size # [mm]
        self.film_size = film_size # [pixel]

        # rendering parameters
        self.mts_Rt = Transformation(R, t) # transformation of the lensgroup
        self.mts_Rt.to(self.device)

        # for visualization
        self.r_last = self.pixel_size * max(self.film_size) / 2

        # TODO: could be further optimized:
        # treat the lenspart as a camera; append one more surface to it
        self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))

        # reverse surfaces
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()
        self.surfaces.reverse()
        self.surfaces.pop(0) # remove sensor plane

        # reverse materials
        self.materials.reverse()

        # aperture plane (TODO: could be optimized further to trace pupil positions)
        self.aperture_radius = self.surfaces[0].r
        self.aperture_distance = self.surfaces[0].d
        self.mts_prepared = True
        self.d_sensor = 0

    def _generate_sensor_samples(self):
        sX, sY = np.meshgrid(
            np.linspace(0, 1, self.film_size[0]),
            np.linspace(0, 1, self.film_size[1])
        )
        return np.stack((sX.flatten(), sY.flatten()), axis=1)

    def _generate_aperture_samples(self):
        Dx = np.random.rand(*self.film_size)
        Dy = np.random.rand(*self.film_size)
        [px, py] = Sampler().concentric_sample_disk(Dx, Dy)
        return np.stack((px.flatten(), py.flatten()), axis=1)

    def sample_ray_sensor_pinhole(self, wavelength, focal_length):
        """
        Sample ray on the sensor plane, assuming a pinhole camera model, given a focal length.
        """
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')
        
        N = np.prod(self.film_size)

        # sensor and aperture plane samplings
        sample2 = self._generate_sensor_samples()
        
        # wavelength [nm]
        wavelength = torch.Tensor(wavelength * np.ones(N))

        # normalized to [-0,5, 0.5]
        sample2 = sample2 - 0.5

        # sample sensor and aperture planes
        p_sensor = sample2 * np.array([
            self.pixel_size * self.film_size[0], self.pixel_size * self.film_size[1]
        ])[None,:]
        
        # aperture samples (last surface plane)
        p_aperture = 0
        d_xy = p_aperture - p_sensor

        # construct ray
        o = torch.Tensor(np.hstack((p_sensor, np.zeros((N,1)))).reshape((N,3)))
        d = torch.Tensor(np.hstack((d_xy, focal_length * np.ones((N,1)))).reshape((N,3)))
        d = normalize(d)

        ray = Ray(o, d, wavelength, device=self.device)
        valid = torch.ones(ray.o[..., 2].shape, device=self.device).bool()
        return valid, ray
    
    def sample_ray_sensor(self, wavelength, offset=np.zeros(2)):
        """
        Sample rays on the sensor plane.
        """
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')
        
        N = np.prod(self.film_size)
        
        # sensor and aperture plane samplings
        sample2 = self._generate_sensor_samples()
        sample3 = self._generate_aperture_samples()

        # wavelength [nm]
        wav = wavelength * np.ones(N)
        
        # sample ray
        valid, ray = self._sample_ray_render(N, wav, sample2, sample3, offset)
        ray_new = self.mts_Rt.transform_ray(ray)
        return valid, ray_new
    
    def _sample_ray_render(self, N, wav, sample2, sample3, offset):
        """
        `offset`: sensor position offsets [mm].
        """

        # sample2 \in [ 0, 1]^2
        # sample3 \in [-1, 1]^2
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')
        
        # normalized to [-0,5, 0.5]
        sample2 = sample2 - 0.5
        
        # sample sensor and aperture planes
        p_sensor = sample2 * np.array([
            self.pixel_size * self.film_size[0], self.pixel_size * self.film_size[1]
        ])[None,:]

        # perturb sensor position by half pixel size
        p_sensor = p_sensor + (np.random.rand(*p_sensor.shape) - 0.5) * self.pixel_size

        # offset sensor positions
        p_sensor = p_sensor + offset

        # aperture samples (last surface plane)
        p_aperture = sample3 * self.aperture_radius
        d_xy = p_aperture - p_sensor

        # construct ray
        o = torch.Tensor(np.hstack((p_sensor, np.zeros((N,1)))).reshape((N,3)))
        d = torch.Tensor(np.hstack((d_xy, self.aperture_distance.item() * np.ones((N,1)))).reshape((N,3)))
        d = normalize(d)
        wavelength = torch.Tensor(wav)

        # trace
        valid, ray = self._trace(Ray(o, d, wavelength, device=self.device))
        return valid, ray

    # ------------------------------------------------------------------------------------
    
    def _refract(self, wi, n, eta, approx=False):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://physics.stackexchange.com/a/436252/104805
        """
        if type(eta) is float:
            eta_ = eta
        else:
            if np.prod(eta.shape) > 1:
                eta_ = eta[..., None]
            else:
                eta_ = eta
        
        cosi = torch.sum(wi * n, axis=-1)

        if approx:
            tmp = 1. - eta**2 * (1. - cosi)
            valid = tmp > 0.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        else:
            cost2 = 1. - (1. - cosi**2) * eta**2
            
            # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
            valid = cost2 > 0.
            cost2 = torch.clamp(cost2, min=1e-8)
            tmp = torch.sqrt(cost2)

            # here we do not have to do normalization because if both wi and n are normalized,
            # then output is also normalized.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        return valid, wt

    def _trace(self, ray, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.surfaces)-1  # last index to stop
        is_forward = (ray.d[..., 2] > 0).all()

        # TODO: Check ray origins to ensure valid ray intersections onto the surfaces
        if is_forward:
            return self._forward_tracing(ray, stop_ind, record)
        else:
            return self._backward_tracing(ray, stop_ind, record)

    def _forward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        
        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])

        valid = torch.ones(dim, device=self.device).bool()
        for i in range(stop_ind+1):
            eta = self.materials[i].ior(wavelength) / self.materials[i+1].ior(wavelength)

            # ray intersecting surface
            valid_o, p = self.surfaces[i].ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = self.surfaces[i].normal(p[..., 0], p[..., 1])
            valid_d, d = self._refract(ray.d, -n, eta)
            
            # check validity
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record: # TODO: make it pythonic ...
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)
            ray.o = p
            ray.d = d
        
        if record:
            return valid, ray, oss
        else:
            return valid, ray
        
    def _backward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        
        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])

        valid = torch.ones(dim, device=ray.o.device).bool()
        for i in np.flip(range(stop_ind+1)):
            surface = self.surfaces[i]
            eta = self.materials[i+1].ior(wavelength) / self.materials[i].ior(wavelength)

            # ray intersecting surface
            valid_o, p = surface.ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = surface.normal(p[..., 0], p[..., 1])
            valid_d, d = self._refract(ray.d, n, eta)  # backward: no need to revert the normal

            # check validity
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record: # TODO: make it pythonic ...
                for os, v, pp in zip(oss, valid.numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, ray, oss
        else:
            return valid, ray

    def _generate_points(self, surface, with_boundary=False):
        R = surface.r
        x = y = torch.linspace(-R, R, surface.APERTURE_SAMPLING, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = surface.surface_with_offset(X, Y)
        valid = surface.is_valid(torch.stack((x,y), axis=-1))

        if with_boundary:
            from scipy import ndimage
            tmp = ndimage.convolve(valid.cpu().numpy().astype('float'), np.array([[0,1,0],[1,0,1],[0,1,0]]))
            boundary = valid.cpu().numpy() & (tmp != 4)
            boundary = boundary[valid.cpu().numpy()].flatten()
        points_local = torch.stack(tuple(v[valid].flatten() for v in [X, Y, Z]), axis=-1)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        if with_boundary:
            return points_world, boundary
        else:
            return points_world


class Surface(PrettyPrinter):
    """
    This is the base class for optical surfaces.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions to be defined in sub-classes.

    Args:
        r: Radius of the aperture (default to be circular, unless specified as square).
        d: Distance of z-direction in global coordinate
        is_square: is the aperture square
        device: Torch device
    """
    def __init__(self, r, d, is_square=False, device=torch.device('cpu')):
        if torch.is_tensor(d):
            self.d = d
        else:
            self.d = torch.Tensor(np.asarray(float(d))).to(device)
        self.is_square = is_square
        self.r = float(r)
        self.device = device

        # There are the parameters controlling the accuracy of ray tracing.
        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6 # in [mm], i.e. 50 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6 # in [mm], i.e. 300 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 257
    
    # === Common methods (must not be overridden)
    def surface_with_offset(self, x, y):
        """
        Returns the z coordinate plus the surface's own distance; useful when drawing the surfaces.
        """
        return self.surface(x, y) + self.d
    
    def normal(self, x, y):
        """
        Returns the 3D normal vector of the surface at 2D coordinate (x,y), in local coordinate.
        """
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, axis=-1))

    def surface_area(self):
        """
        Computes the surface's area.
        """
        if self.is_square:
            return self.r**2
        else: # is round
            return np.pi * self.r**2
    
    def mesh(self):
        """
        Generates a 2D meshgrid for the current surface.
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )
        valid_map = self.is_valid(torch.stack((x,y), axis=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, p):
        """
        (Approximated) Signed Distance Function (SDF) of a 2D point p to the surface's aperture boundary.
        If:
        - Returns < 0: p is within the surface's aperture.
        - Returns = 0: p is at the surface's aperture boundary.
        - Returns > 0: p is outside of the surface's aperture.

        Args:
            p: Local 2D point.
        
        Returns:
            A SDF mask.
        """
        if self.is_square:
            return torch.max(torch.abs(p) - self.r, axis=-1)[0]
        else: # is round
            return length2(p) - self.r**2
    
    def is_valid(self, p):
        """
        If a 2D point p is valid, i.e. if p is within the surface's aperture.
        """
        return (self.sdf_approx(p) < 0.0).bool()

    def ray_surface_intersection(self, ray, active=None):
        """
        Computes ray-surface intersection, one of the most crucial functions in this class.
        Given ray(s) and an activity mask, the function computes the intersection point(s),
        and determines if the intersection is valid, and update the active mask accordingly. 
        
        Args:
            ray: Rays.
            active: The initial active mask.

        Returns:
            valid_o: The updated active mask (if the current ray is physically active in tracing).
            local: The computed intersection point(s).
        """
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)
        
        valid_o = solution_found & self.is_valid(local[...,0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, local

    def newtons_method(self, maxt, o, D, option='implicit'):
        """
        Newton's method to find the root of the ray-surface intersection point.
        
        Two modes are supported here:

        1. 'explicit": This implements the loop using autodiff, and gradients will be
        accurate for o, D, and self.parameters. Slow and memory-consuming.
        
        2. 'implicit": This implements the loop using implicit-layer theory, find the 
        solution without autodiff, then hook up the gradient. Less memory-consuming.

        Args:
            maxt: The maximum travel distance of a single ray.
            o: The origins of the rays.
            D: The directional vector of the rays.
            option: The computing modes.

        Returns:
            valid: The updated active mask (if the current ray is physically active in tracing).
            p: The computed intersection point(s).
        """
        
        # pre-compute constants
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = dx**2 + dy**2
        B = 2 * (dx * ox + dy * oy)
        C = ox**2 + oy**2

        # initial guess of t
        t0 = (self.d - oz) / dz
        
        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
            )
        elif option == 'implicit':
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(
                    maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
                )
                s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                    t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
                )[1]
            t = t0 + t_delta
            t = t - (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d)/s_derivatives_dot_D
        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t[..., None] * D

        return valid, p

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        """
        The actual implementation of Newton's method.

        Args:
            dx,dy,dx,ox,oy,oz,A,B,C: Variables to a quadratic problem.
        
        Returns:
            t: The travel distance of the ray.
            t_delta: The incremental change of t at each iteration.
            valid: The updated active mask (if the current ray is physically active in tracing).
        """
        if oz.numel() < 2:
            oz = torch.Tensor([oz.item()]).to(self.device)
        t_delta = torch.zeros_like(oz)

        # iterate until the intersection error is small
        t        = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t = t0 + t_delta
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C # here z = t_delta * dz
            )
            t_delta = t_delta - residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    # === Virtual methods (must be overridden)
    def g(self, x, y):
        """
        Function g(x,y).

        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            g(x,y): Function g(x,y) at (x,y).
        """
        raise NotImplementedError()

    def dgd(self, x, y):
        """
        Derivatives of g: (dg/dx, dg/dy).

        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            dg/dx: dg/dx of function g(x,y) at (x,y).
            dg/dy: dg/dy of function g(x,y) at (x,y).
        """
        raise NotImplementedError()

    def h(self, z):
        """
        Function h(z).

        Args:
            z: The z local coordinate.

            
        Returns:
            h(z): Function h(z) at z.
        """
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivatives of h: dh/dz.

        Args:
            z: The z local coordinate.

        Returns:
            dh/dz: dh/dz of function h(z) at z.
        """
        raise NotImplementedError()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        
        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            z: Surface's z coordinate.
        """
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    # === Default methods (better be overridden)
    def surface_derivatives(self, x, y):
        """
        Computes the surface's spatial derivatives:
        
        Assume the surface height function f(x,y,z) = g(x,y) + h(z). The spatial derivatives are:
        
        \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        
        (Note: this default implementation is not efficient)
        
        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            gx: dg/dx.
            gy: dg/dy.
            hz: dh/dz.
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)
        
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        """
        Computes the surface and the dot product of its spatial derivatives and ray direction.

        Assume the surface height function f(x,y,z) = g(x,y) + h(z). The outputs are:
        
        g(x,y) + h(z)  and  (dg/dx, dg/dy, dh/dz) \cdot (dx,dy,dz).

        (Note: this default implementation is not efficient)

        Args:
            t: The travel distance of the considered ray(s).
            dx,dy,dx,ox,oy,oz,A,B,C: Variables to a quadratic problem.

        Returns:
            s: Value of f(x,y,z). The intersection is at the surface if s equals zero.
            sx*dx + sy*dy + sz*dz: The dot product between the surface's spatial derivatives and ray direction d.
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x,y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx*dx + sy*dy + sz*dz


class Aspheric(Surface):
    """
    This is the aspheric surface class, implementation follows: https://en.wikipedia.org/wiki/Aspheric_lens.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2/R**2 )) + ai[0] * r**4 + ai[1] * r**6 + \cdots.
    h(z) = -z.
    
    Args (new attributes):
        c: Surface curvature, or one over radius of curvature.
        k: Conic coefficient.
        ai: Aspheric parameters, could be a vector. When None, the surface is spherical.
    """
    def __init__(self, r, d, c=0., k=0., ai=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.c, self.k = (torch.Tensor(np.array(v)) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))

    # === Common methods
    def g(self, x, y):
        return self._g(x**2 + y**2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2*x, dsdr2*y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x**2 + y**2)

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2*x, dsdr2*y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        #pylint: disable=unused-argument
        # TODO: could be further optimized
        r2 = A * t**2 + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2*A*t + B) - dz
    
    # === Private methods
    def _g(self, r2):
        tmp = r2*self.c
        total_surface = tmp / (1 + torch.sqrt(1 - (1+self.k) * tmp*self.c))
        higher_surface = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2**2
        return total_surface + higher_surface
    
    def _dgd(self, r2):
        alpha_r2 = (1 + self.k) * self.c**2 * r2
        tmp = torch.sqrt(1 - alpha_r2) # TODO: potential NaN grad
        total_derivative = self.c * (1 + tmp - 0.5*alpha_r2) / (tmp * (1 + tmp)**2)

        higher_derivative = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * higher_derivative + (i+2) * self.ai[i]
        return total_derivative + higher_derivative * r2


# ----------------------------------------------------------------------------------------

class BSpline(Surface):
    """
    This is the B-Spline surface class, implementation follows Wikipedia, for freeform surfaces.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) is parameterized by a B-Spline surface.
    h(z) = -z.
    
    Args (new attributes):
        px: Polynomial order in x direction.
        py: Polynomial order in y direction.
        tx: Knots in x.
        ty: Knots in y.
        c: Spline coefficients.
    """
    def __init__(self, r, d, size, px=3, py=3, tx=None, ty=None, c=None, is_square=False, device=torch.device('cpu')): # input c is 1D
        Surface.__init__(self, r, d, is_square, device)
        self.px = px
        self.py = py
        self.size = np.asarray(size)

        # knots
        if tx is None:
            self.tx = None
        else:
            if len(tx) != size[0] + 2*(self.px + 1):
                raise Exception('len(tx) is not correct!')
            self.tx = torch.Tensor(np.asarray(tx)).to(self.device)
        if ty is None:
            self.ty = None
        else:
            if len(ty) != size[1] + 2*(self.py + 1):
                raise Exception('len(ty) is not correct!')
            self.ty = torch.Tensor(np.asarray(ty)).to(self.device)

        # c is the only differentiable parameter
        c_shape = size + np.array([self.px, self.py]) + 1
        if c is None:
            self.c = None
        else:
            c = np.asarray(c)
            if c.size != np.prod(c_shape):
                raise Exception('len(c) is not correct!')
            self.c = torch.Tensor(c.reshape(*c_shape)).to(self.device)
        
        if (self.tx is None) or (self.ty is None) or (self.c is None):
            self.tx = self._generate_knots(self.r, size[0], p=px, device=device)
            self.ty = self._generate_knots(self.r, size[1], p=py, device=device)
            self.c = torch.zeros(*c_shape, device=device)
        else:
            self.to(self.device)

    @staticmethod
    def _generate_knots(R, n, p=3, device=torch.device('cpu')):
        t = np.linspace(-R, R, n)
        step = t[1] - t[0]
        T = t[0] - 0.9 * step
        np.pad(t, p+1, 'constant', constant_values=step)
        t = np.concatenate((np.ones(p+1)*T, t, -np.ones(p+1)*T), axis=0)
        return torch.Tensor(t).to(device)

    def fit(self, x, y, z, eps=1e-3):
        x, y, z = (v.flatten() for v in [x, y, z])

        # knot positions within [-r, r]^2
        X = np.linspace(-self.r, self.r, self.size[0])
        Y = np.linspace(-self.r, self.r, self.size[1])
        bs = LSQBivariateSpline(x, y, z, X, Y, kx=self.px, ky=self.py, eps=eps)
        # print('RMS residual error is {} um'.format(np.sqrt(bs.fp/len(z))*1e3))
        tx, ty = bs.get_knots()
        c = bs.get_coeffs().reshape(len(tx)-self.px-1, len(ty)-self.py-1)

        # convert to torch.Tensor
        self.tx, self.ty, self.c = (torch.Tensor(v).to(self.device) for v in [tx, ty, c])

    # === Common methods
    def g(self, x, y):
        return self._deBoor2(x, y)

    def dgd(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1)

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._deBoor2(x, y)

    def surface_derivatives(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1), -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        #pylint: disable=unused-argument
        x = ox + t * dx
        y = oy + t * dy
        s, sx, sy = self._deBoor2(x, y, dx=-1, dy=-1)
        return s - z, sx*dx + sy*dy - dz

    def reverse(self):
        self.c = -self.c

    # === Private methods
    def _deBoor(self, x, t, c, p=3, is2Dfinal=False, dx=0):
        """
        Arguments
        ---------
        x: Position.
        t: Array of knot positions, needs to be padded as described above.
        c: Array of control points.
        p: Degree of B-spline.
        dx:
        - 0: surface only
        - 1: surface 1st derivative only
        - -1: surface and its 1st derivative
        """
        k = torch.sum((x[None,...] > t[...,None]).int(), axis=0) - (p+1)
        
        if is2Dfinal:
            inds = np.indices(k.shape)[0]
            def _c(jk): return c[jk, inds]
        else:
            def _c(jk): return c[jk, ...]

        need_newdim = (len(c.shape) > 1) & (not is2Dfinal)

        def f(a, b, alpha):
            if need_newdim:
                alpha = alpha[...,None]
            return (1.0 - alpha) * a + alpha * b
        
        # surface only
        if dx == 0:
            d = [_c(j+k) for j in range(0, p+1)]

            for r in range(-p, 0):
                for j in range(p, p+r, -1):
                    left = j+k
                    t_left  = t[left]
                    t_right = t[left-r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j-1], d[j], alpha)
            return d[p]

        # surface 1st derivative only
        if dx == 1:
            q = []
            for j in range(1, p+1):
                jk = j+k
                tmp = t[jk+p] - t[jk]
                if need_newdim:
                    tmp = tmp[..., None]
                q.append(p * (_c(jk) - _c(jk-1)) / tmp)

            for r in range(-p, -1):
                for j in range(p-1, p+r, -1):
                    left = j+k
                    t_right = t[left-r]
                    t_left_ = t[left+1]
                    alpha = (x - t_left_) / (t_right - t_left_)
                    q[j] = f(q[j-1], q[j], alpha)
            return q[p-1]
            
        # surface and its derivative (all)
        if dx < 0:
            d, q = [], []
            for j in range(0, p+1):
                jk = j+k
                c_jk = _c(jk)
                d.append(c_jk)
                if j > 0:
                    tmp = t[jk+p] - t[jk]
                    if need_newdim:
                        tmp = tmp[..., None]
                    q.append(p * (c_jk - _c(jk-1)) / tmp)

            for r in range(-p, 0):
                for j in range(p, p+r, -1):
                    left = j+k
                    t_left  = t[left]
                    t_right = t[left-r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j-1], d[j], alpha)

                    if (r < -1) & (j < p):
                        t_left_ = t[left+1]
                        alpha = (x - t_left_) / (t_right - t_left_)
                        q[j] = f(q[j-1], q[j], alpha)
            return d[p], q[p-1]

    def _deBoor2(self, x, y, dx=0, dy=0):
        """
        Arguments
        ---------
        x,  y : Position.
        dx, dy: 
        """
        if not torch.is_tensor(x):
            x = torch.Tensor(np.asarray(x)).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(np.asarray(y)).to(self.device)
        dim = x.shape

        x = x.flatten()
        y = y.flatten()

        # handle boundary issue
        x = torch.clamp(x, min=-self.r, max=self.r)
        y = torch.clamp(y, min=-self.r, max=self.r)

        if (dx == 0) & (dy == 0):     # spline
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s = self._deBoor(y, self.ty, s_tmp.T, self.py, True)
            return s.reshape(dim)
        elif (dx == 1) & (dy == 0):  # x-derivative
            s_tmp = self._deBoor(y, self.ty, self.c.T, self.py)
            s_x = self._deBoor(x, self.tx, s_tmp.T, self.px, True, dx)
            return s_x.reshape(dim)
        elif (dy == 1) & (dx == 0):  # y-derivative
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s_y = self._deBoor(y, self.ty, s_tmp.T, self.py, True, dy)
            return s_y.reshape(dim)
        else:                       # return all
            s_tmpx = self._deBoor(x, self.tx, self.c, self.px)
            s_tmpy = self._deBoor(y, self.ty, self.c.T, self.py)
            s, s_x = self._deBoor(x, self.tx, s_tmpy.T, self.px, True, -abs(dx))
            s_y = self._deBoor(y, self.ty, s_tmpx.T, self.py, True, abs(dy))
            return s.reshape(dim), s_x.reshape(dim), s_y.reshape(dim)


class XYPolynomial(Surface):
    """
    This is the XY polynomial surface class, for freeform surfaces.
    
    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) = \sum{i,j} a_ij x^i y^{j-i}.
    h(z) = b z^2 - z.

    Or, re-write f(x,y,z) in the following:

    explicit:   b z^2 - z + \sum{i,j} a_ij x^i y^{j-i} = 0
    implicit:   (denote c = \sum{i,j} a_ij x^i y^{j-i})
                z = (1 - \sqrt{1 - 4 b c}) / (2b)
                
    explicit derivatives:
    (2 b z - 1) dz + \sum{i,j} a_ij x^{i-1} y^{j-i-1} ( i y dx + (j-i) x dy ) = 0

    dx = \sum{i,j} a_ij   i   x^{i-1} y^{j-i}
    dy = \sum{i,j} a_ij (j-i) x^{i}   y^{j-i-1}
    dz = 2 b z - 1
    
    Args (new attributes):
        J: Polynomial order.
        ai: Polynomial coefficients.
        b: Coefficient in h(z).
    """
    def __init__(self, r, d, J=0, ai=None, b=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.J  = J
        # differentiable parameters (default: all ai's and b are zeros)
        if ai is None:
            self.ai = torch.zeros(self.J2aisize(J)) if J > 0 else torch.array([0])
        else:
            if len(ai) != self.J2aisize(J):
                raise Exception("len(ai) != (J+1)*(J+2)/2 !")
            self.ai = torch.Tensor(ai).to(device)
        if b is None:
            b = 0.
        self.b = torch.Tensor(np.asarray(b)).to(device)
        print('ai.size = {}'.format(self.ai.shape[0]))
        self.to(self.device)
    
    @staticmethod
    def J2aisize(J):
        return int((J+1)*(J+2)/2)

    def center(self):
        x0 = -self.ai[2]/self.ai[5]
        y0 = -self.ai[1]/self.ai[3]
        return x0, y0

    def fit(self, x, y, z):
        x, y, z = (torch.Tensor(v.flatten()) for v in [x, y, z])
        A, AT = self._construct_A(x, y, z**2)
        coeffs = torch.solve(AT @ z[...,None], AT @ A)[0]
        self.b  = coeffs[0][0]
        self.ai = coeffs[1:].flatten()

    # === Common methods
    def g(self, x, y):
        if type(x) is torch.Tensor:
            c = torch.zeros_like(x)
        elif type(x) is np.ndarray:
            c = np.zeros_like(x)
        else:
            c = 0.0
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                # c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                c = c + self.ai[count] * x**i * y**(j-i)
                count += 1
        return c

    def dgd(self, x, y):
        if type(x) is torch.Tensor:
            sx = torch.zeros_like(x)
            sy = torch.zeros_like(x)
        elif type(x) is np.ndarray:
            sx = np.zeros_like(x)
            sy = np.zeros_like(x)
        else:
            sx = 0.0
            sy = 0.0
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                if j > 0:
                    sx = sx + self.ai[count] * i * x**max(i-1,0) * y**(j-i)
                    sy = sy + self.ai[count] * (j-i) * x**i * y**max(j-i-1,0)
                count += 1
        return sx, sy

    def h(self, z):
        return self.b * z**2 - z

    def dhd(self, z):
        return 2 * self.b * z - torch.ones_like(z)

    def surface(self, x, y):
        c = self.g(x, y)
        return self._solve_for_z(c)

    def reverse(self):
        self.b = -self.b
        self.ai = -self.ai

    def surface_derivatives(self, x, y):
        x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
                    sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
                count += 1
        z = self._solve_for_z(c)
        return sx, sy, self.dhd(z)
        
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        #pylint: disable=unused-argument
        # (basically a copy of `surface_derivatives`)
        x = ox + t * dx
        y = oy + t * dy
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
                    sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
                count += 1
        s = c + self.h(z)
        return s, sx*dx + sy*dy + self.dhd(z)*dz

    # === Private methods
    def _construct_A(self, x, y, A_init=None):
        A = torch.zeros_like(x) if A_init == None else A_init
        for j in range(self.J+1):
            for i in range(j+1):
                A = torch.vstack((A, torch.pow(x, i) * torch.pow(y, j-i)))
        AT = A[1:,:] if A_init == None else A
        return AT.T, AT

    def _solve_for_z(self, c):
        # TODO: potential NaN grad
        if self.b == 0:
            return c
        else:
            return (1. - torch.sqrt(1. - 4*self.b*c)) / (2*self.b)


class Mesh(Surface):
    """
    This is the linear mesh surface class, for freeform surfaces.
    
    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) is parameterized by its attribute c.
    h(z) = z.

    Args (new attributes):
        c: Surface rasterization array, i.e. the 2D discretization for the surface's height.
    """
    def __init__(self, r, d, size, c=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        if c is None:
            self.c = torch.zeros(size).to(device)
        else:
            c_shape = size + np.array([self.px, self.py]) + 1
            c = np.asarray(c)
            if c.size != np.prod(c_shape):
                raise Exception('len(c) is not correct!')
            self.c = torch.Tensor(c.reshape(*c_shape)).to(device)
        self.size = torch.Tensor(np.array(size)) # screen image dimension [pixel]
        self.size_np = size # screen image dimension [pixel]

    # === Common methods
    def g(self, x, y):
        return self._shading(x, y)
    
    def dgd(self, x, y):
        p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
        p_floor = torch.floor(p).long()
        x0, y0 = p_floor[...,0], p_floor[...,1]
        s00, s01, s10, s11 = self._tex4(x0, y0)
        denominator = 2 * (2*self.r / self.size)
        return (s10 - s00 + s11 - s01) / denominator[0], (s01 - s00 + s11 - s10) / denominator[1]

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._shading(x, y)

    def surface_derivatives(self, x, y):
        sx, sy = self.dgd(x, y)
        return sx, sy, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        #pylint: disable=unused-argument
        x = ox + t * dx
        y = oy + t * dy
        
        p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
        p_floor = torch.floor(p).long()

        # linear interpolation
        x0, y0 = p_floor[...,0], p_floor[...,1]
        s00, s01, s10, s11 = self._tex4(x0, y0)
        w1 = p - p_floor
        w0 = 1. - w1
        s = (
            w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
            w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
        )
        denominator = 2 * (2*self.r / (self.size-1))
        sx = (s10 - s00 + s11 - s01) / denominator[0]
        sy = (s01 - s00 + s11 - s10) / denominator[1]
        return s - z, sx*dx + sy*dy - dz

    def reverse(self):
        self.c = -self.c

    # === Private methods
    def _tex(self, x0, y0, bmode=BoundaryMode.replicate): # texture indexing four pixels
        return tex(self.c, self.size_np, x0, y0, bmode)
    
    def _tex4(self, x0, y0, bmode=BoundaryMode.replicate): # texture indexing four pixels
        return tex4(self.c, self.size_np, x0, y0, bmode)
    
    def _shading(self, x, y, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
        p_floor = torch.floor(p).long()

        if lmode is InterpolationMode.nearest:
            val = self._tex(p_floor[...,0], p_floor[...,1], bmode)
        elif lmode is InterpolationMode.linear:
            x0, y0 = p_floor[...,0], p_floor[...,1]
            s00, s01, s10, s11 = self._tex4(x0, y0, bmode)
            w1 = p - p_floor
            w0 = 1. - w1
            val = (
                w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
                w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
            )
        return val
