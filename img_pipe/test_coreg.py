
import nipy
import nipy.algorithms.registration.histogram_registration
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

mriimg = nipy.load_image('C:\data\subjects\patient_25\mri\orig.mgz')
ctimg = nipy.load_image('C:\data\subjects\patient_25\CT\CT.nii')

ct_cmap = ctimg.coordmap

ct_to_mri_reg = nipy.algorithms.registration.histogram_registration.HistogramRegistration(ctimg, mriimg, similarity='nmi', smooth=0., interp='pv')
aff = ct_to_mri_reg.optimize('rigid').as_affine()

# Get affine transform 
affine = mriimg.affine
fsVox2RAS = np.array([[-1., 0., 0., 128.], 
                            [0., 0., 1., -128.], 
                            [0., -1., 0., 128.], 
                            [0., 0., 0., 1.]])

# Apply orientation to the MRI so that the order of the dimensions will be
# sagittal, coronal, axial
codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(affine))
img_data = nib.orientations.apply_orientation(mriimg.get_data(), codes)
voxel_sizes = nib.affines.voxel_sizes(affine)
nx,ny,nz = np.array(img_data.shape, dtype='float')

inv_affine = np.linalg.inv(affine)
img_clim = np.percentile(img_data, (1., 99.))

## Apply orientation to pial surface fill
#self.pial_codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(self.pial_img.affine))
#pial_data = nib.orientations.apply_orientation(self.pial_img.get_data(), self.pial_codes)
#pial_data = scipy.ndimage.binary_closing(pial_data)

# Apply orientation to the CT so that the order of the dimensions will be
# sagittal, coronal, axial
ct_codes =nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(aff))
ct_data = nib.orientations.apply_orientation((ctimg).get_data(), aff)
print('Done')
