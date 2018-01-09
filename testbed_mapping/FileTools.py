import h5py
import numpy as np
import healpy as hp

def WriteH5Py(filename, datadict, mode='w'):
    """
    """

    f = h5py.File(filename, mode)

    for k, v in datadict.items():
        dset = f.create_dataset(k, v.shape, dtype=v.dtype)
        dset[...] = v[...]

    f.close()

def AppendH5Py(filename, data, key, ranges, mode='a'):
    """
    ranges = (range(i1,i2), range(j1,j2), ....)
    """

    f = h5py.File(filename, mode)

    #print data.shape
    for i in ranges[1]:
        print(i,f[key][ranges[0],i].shape)
        f[key][ranges[0],i] = data[:,i]

    f.close()
    
def ReadH5Py(filename, keys=None):
    """
    """

    f = h5py.File(filename, 'r')
        

    if isinstance(keys, type(None)):
        dataset = {str(key): f[key][...] for key in f.keys() if key != 'HEADER'}
    else:
        dataset = {str(keys[0]): f[keys[0]][...]}

    if 'HEADER' in f.keys():
        dataset['HEADER'] = {str(key) : f['HEADER'].attrs[key] for key in f['HEADER'].attrs.keys()} 
    f.close()

    return dataset



def ReadMaps(infodict, mode, nchannels, npix_out):
    """
    Reads in a HEALPix map and converts it to the require pixelisation
    of the simulation.
    """

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    out_maps = np.zeros((nchannels, npix_out))
    channels  = np.arange(nchannels, dtype='int')

    if rank == 0:
        if infodict['Inputs']['sim_{}'.format(mode)]:
            # First we read in the data map
            maps = hp.read_map('inputs/{}/{}_Model1.fits'.format(mode, mode), channels)
            m_nside = int(np.sqrt(maps[0].size/12.))
            
            
            if infodict['MapMaking']['mm_pixelisation'] == 'HEALPix':
                # If the out put map is in HEALPix form ...
                nside_out = int(np.sqrt(npix_out/12.))
                for i, m in enumerate(maps):
                    if m_nside != nside_out:
                        out_maps[i, :] = hp.ud_grade(m, nside_out)
                    else:
                        out_maps[i, :] = m
            elif infodict['MapMaking']['mm_pixelisation'] == 'Cartesian':
                # If the output map is Cartesian form ...
                wcs = CartPix.Info2WCS(infodict)
                # First get coordinates of each cartesian pixel
                xpix, ypix = np.meshgrid(np.arange(0, infodict['MapMaking']['mm_cart_xpix']),
                                         np.arange(0, infodict['MapMaking']['mm_cart_ypix']))
                c_phi, c_theta = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(), 0)
                
                # Now get the data for the healpix map
                pix = hp.ang2pix(m_nside, (90.-c_theta)*np.pi/180., c_phi*np.pi/180.)

                for i, m in enumerate(maps):
                    out_maps[i, :] = m[pix]

    comm.barrier()
    comm.Bcast([out_maps, MPI.FLOAT], root=0)
    return out_maps
