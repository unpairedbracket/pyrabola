import numpy as np

class Propagator:
    pass

class Hermite(Propagator):
    @staticmethod
    def generate(x,bandlimit=None):
        # This generates Hermite functions ψn(x / sqrt(2))
        # Which are the eigenfunctions of the focusing envelope equation

        dx = np.diff(x.flat).mean()

        if not bandlimit:
            bandlimit = x.size
        elif bandlimit == 'conservative':
            # Bandlimit to keep functions within bounds of x
            bandlimit = np.max(x**2)
        elif bandlimit == 'fourier':
            bandlimit = (np.pi / (2 * dx))**2

        n_func = np.ceil(bandlimit).astype('int')

        V = np.zeros((n_func, x.size), 'longdouble')
        x = x.astype('longdouble')

        V[0,:] = np.exp(700-x**2)
        V[1,:] = np.exp(700-x**2) * 2 * x

        for j in np.arange(1, n_func-1):
            V[j+1, :] = np.sqrt(1 / (j+1)) * (2 * x * V[j, :] - np.sqrt(j) * V[j-1,:])

        return V * (2 / np.pi)**0.25 * np.exp(-700)

    def __init__(self, x, y, bandlimit=None):
        self.x, self.y = x, y
        self.dx = np.diff(x).mean()
        self.dy = np.diff(y).mean()
        if type(bandlimit) is not tuple:
            bandlimit = (bandlimit, bandlimit)
        self.U = Hermite.generate(x, bandlimit[0])
        self.V = Hermite.generate(y, bandlimit[1])
        self.bandlimit = (self.U.shape[0], self.V.shape[0])

        self.mode_numbers = (
                np.arange(self.bandlimit[0])[:, None]
              + np.arange(self.bandlimit[1])[None, :]
        )

    def copy(self):
        ''' TODO: actually copy'''
        return self

    def decompose(self, A):
        return self.U @ A @ self.V.T * self.dx * self.dy

    def get_phases(self, dphi):
        return np.exp(-1j * self.mode_numbers * dphi)

    def recompose(self, coeffs):
        return self.U.T @ coeffs @ self.V


    def propagate_iter(self, A, psi, psi0=-np.pi/2):
        # x: (n_x,) 1d x axis
        # y: (n_y,) 1d y axis
        # A: (n_x, n_y) 2d complex field amplitude. Note 'ij' meshgrid indexing.
        # psi: (n_psi,) points to evaluate A(psi) at

        coeffs = self.decompose(A)

        for i, p in enumerate(psi):
            A_p = self.recompose(coeffs * self.get_phases(p - psi0))
            print(i)
            yield A_p

    def propagate_slices(self, A, psi, slices, psi0=-np.pi/2, bandlimit=None):
        A_all_sliced = [
            np.zeros(psi.shape + A[slc].shape, dtype='complex') for slc in slices
        ]
        for n, A in enumerate(self.propagate_iter(A, psi, psi0, bandlimit)):
            for A_sliced, slc in zip(A_all_sliced, slices):
                A_sliced[n] = A[slc]

        return A_all_sliced

    def propagate_to(self, A, psi, psi0=-np.pi/2, bandlimit=None):
        # x: (n_x,) 1d x axis
        # y: (n_y,) 1d y axis
        # A: (n_x, n_y) 2d complex field amplitude. Note 'ij' meshgrid indexing.
        # psi: (n_psi,) points to evaluate A(psi) at
         
        return tuple(self.propagate_iter(A, psi, psi0, bandlimit))

