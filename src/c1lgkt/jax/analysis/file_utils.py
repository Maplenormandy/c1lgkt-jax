"""
Some utility scripts to load files
"""

import re
import numpy as np

from jaxtyping import Real, ArrayLike



def load_pfile(filename: str) -> dict:
    p = {}
    with open(filename) as f:
        data = f.readlines()

        # Start with the first line
        tokens = data[0].split()
        # Get the number of points
        p['Npsi'] = int(tokens[0])

        # We assume that the rest of the tokens always follow a uniform format
        def read_array(line: int) -> int:
            # This function reads the appropriate lines and returns the next line to read

            # Read the header line
            header_tokens = data[line].split()

            # number of lines to read
            nlines = int(header_tokens[0])

            # Arrays to accumulate grid values
            arr = np.empty((nlines, 3))

            # Read the next nlines lines to get the array values
            for i in range(nlines):
                tokens = data[line + 1 + i].split()
                arr[i, 0] = float(tokens[0])
                arr[i, 1] = float(tokens[1])
                arr[i, 2] = float(tokens[2])

            # For now, we only process radial fields
            if header_tokens[1] == 'psinorm':
                field = header_tokens[2].split('(')[0]
                dfield = header_tokens[3]
                p['psinorm'] = arr[:, 0]
                p[field] = arr[:, 1]
                p[dfield] = arr[:, 2]

            # Return the next line to read and the array
            return line + 1 + nlines
        
        # Start again with the first line
        line = 0
        while line < len(data):
            if data[line].isspace():
                # Skip blank lines
                line += 1
                continue
            else:
                # Read the array and update the line number
                line = read_array(line)

    return p


def load_gfile(filename: str) -> dict:
    """
    Loads a gfile and returns a dictionary containing the relevant data. Adapted from markchil/eqtools
    """
    # Dictionary to hold the gfile data
    g = {}

    with open(filename, 'r') as f:
        data = f.readlines()

        # Start with the first line
        tokens = data[0].split()
        # Get the number of radial and vertical grid points
        g['Nr'] = int(tokens[-2])
        g['Nz'] = int(tokens[-1])
        g['Npsi'] = g['Nr']

        # Helper function to read a line of tokens
        def read_tokens(line: int):
            return list(map(float,re.findall(r'-?\d\.\d*[eE][-+]\d*', data[line])))

        # The second line contains information for constructing RZ grid
        g['rdim'], g['zdim'], g['rcentr'], g['rmin'], g['zmid'] = read_tokens(line=1)
        g['rmax'] = g['rmin'] + g['rdim']
        g['zmin'] = g['zmid'] - g['zdim'] / 2
        g['zmax'] = g['zmid'] + g['zdim'] / 2

        g['rgrid'] = np.linspace(g['rmin'], g['rmax'], g['Nr'])
        g['zgrid'] = np.linspace(g['zmin'], g['zmax'], g['Nz'])

        # The third line contains R,Z of magnetic axis, psi at magnetic axis, and LCFS
        g['raxis'], g['zaxis'], g['psiaxis'], g['psix'], g['bcentr'] = tokens = read_tokens(line=2)

        # Out of convenience, renormalize psi such that psiaxis = 0
        g['psix'] -= g['psiaxis']
        
        # read EFIT-calculated plasma current, psi at magnetic axis (duplicate),
        # dummy, R of magnetic axis (duplicate), dummy
        g['ip'], _, _, _, _ = read_tokens(line=3)

        # Skip the 5th line
        _, _, _, _, _ = read_tokens(line=4)

        # Start keeping track of the current line
        line = 5

        # Helper function to read arrays
        def read_array(begin_read: int, npts: int):
            # Number of rows to read in an array
            nrows = npts//5
            if npts % 5 != 0:     # catch truncated rows
                nrows += 1

            temp_array = []
            for i in range(nrows):
                temp_array.extend(read_tokens(line=begin_read + i))
            return begin_read + nrows, np.array(temp_array)
        
        # First, read in ff
        line, g['ff'] = read_array(line, g['Npsi'])
        # NOTE: sign convention for ff in gfile is opposite to the one we use, so flip the sign here.
        g['ff'] = -g['ff']
        # Next, read pressure
        line, g['fluxPres'] = read_array(line, g['Npsi'])
        # Read ffprim
        line, g['ffprim'] = read_array(line, g['Npsi'])
        # Read pprime
        line, g['pprime'] = read_array(line, g['Npsi'])

        # psi grid on which the flux functions are defined
        g['psi'] = np.linspace(0, g['psix'], g['Npsi'])

        # Now, read the 2d psirz array
        line, g['psirz'] = read_array(line, g['Nr'] * g['Nz'])
        g['psirz'] = g['psirz'].reshape((g['Nz'], g['Nr'])) - g['psiaxis']  # renormalize psirz

        # Now read q profile
        line, g['qpsi'] = read_array(line, g['Npsi'])

        # Now, we read the LCFS and wall points
        tokens = data[line].split()
        g['Nlcfs'] = int(tokens[0])
        g['Nwall'] = int(tokens[1])
        line += 1

        line, g['lcfsrz'] = read_array(line, 2*g['Nlcfs'])
        g['lcfsrz'] = g['lcfsrz'].reshape((g['Nlcfs'], 2)).T
        line, g['wallrz'] = read_array(line, 2*g['Nwall'])
        g['wallrz'] = g['wallrz'].reshape((g['Nwall'], 2)).T

        # Estimate the X-point location from the LCFS points.
        # TODO: Need to do something about multiple X-points
        zmin_idx = np.argmin(g['lcfsrz'][1,:])
        g['zx'] = g['lcfsrz'][1, zmin_idx]
        g['rx'] = g['lcfsrz'][0, zmin_idx]
    
    return g