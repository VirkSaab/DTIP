/*============================================================================

  Program:     DTI ToolKit (DTI-TK)
  Module:      $RCSfile: Volume.h,v $
  Language:    C++
  Date:        $Date: 2012/08/01 17:06:59 $
  Version:     $Revision: 1.2 $

  Copyright (c) Gary Hui Zhang (garyhuizhang@gmail.com).
  All rights reserverd.

  DTI-TK is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  DTI-TK is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with DTI-TK.  If not, see <http://www.gnu.org/licenses/>.
============================================================================*/


// template Volume<class Object>
//
// declaration and implementation
//
// With only forward declarations of VTKReader, VTKWriter
// the compilation will fail.

#ifndef _volume_Volume_H
#define _volume_Volume_H

#include "VoxelSpace.h"
#include "../geometry/Vector3D.h"
#include "../geometry/SymTensor3D.h"
#include "../geometry/Reflection3D.h"
#include "nifti1_io.h"
#include "../io/VTKReader.h"
#include "../io/VTKWriter.h"
#include "../io/util.h"
#include <iostream>
#include <iomanip>
#include <ctime>

namespace volume {
	
	using namespace geometry;
	using namespace io;
	using namespace std;
	
	//declaration	
	template <class Object>
	class Volume : public VoxelSpace {
		
		typedef io::VTKReader<Object> VTKReader;
		typedef io::VTKWriter<Object> VTKWriter;
		
		public:
		Object ***voxel;
		
		// gradient info
		Object ***grad[3];
				
		protected:
		// symbolic zero
		// default value for background for all objects
		// need/can only be initialized from extended classes!!!
		Object zero;
		Object bg;
				
		// gradient enabled flag
		bool gradEnabled;
		
		// persistent local variables
	 	// cornersIndex:
		// index of four corners of a square for interpolation
		// in the order of z, y, x
		// for example:
		// cornersIndex[0][1][0][0] and cornersIndex[0][1][0][1] are
		// the x and y indices for the corner (0,1,0)
		//
		// corners:
		// corners					coordinate system
		// [Z][0][0] -- bottom left		(0,0,z)
		// [Z][0][1] -- bottom right		(1,0,z)
		// [Z][1][0] -- top left			(0,1,z)
		// [Z][1][1] -- top right		(1,1,z)
		// [Z][Y][X]					(x,y,z)
		//
		// lambda[0] is the lambda along x
		// lambda[1] is the lambda along y
		// lambda[2] is the lambda along z
		// 
		// bottomLeft[0] is the x index
		// bottomLeft[1] is the y index
		// bottomLeft[2] is the z index
		#define _COMMON_VAR int cornerIndex[2][2][2][3]; \
			const Object *corner[2][2][2]; double lambda[3]; \
			int bottomLeft[3];
		
		protected:
		// default constructor with no information regarding
		// xsize, ysize etc.  no allocation is possible
		Volume (bool enableGrad = false) : VoxelSpace() {
			voxel = NULL;
			gradEnabled = enableGrad;
			for (int i = 0; i < 3; ++i) {
				grad[i] = NULL;
			}
		}
		
		// data type specific mapping to LPI
		virtual void objectSpecificOrientationMapping (const int mapping[3], const bool dir[3]) {
			// default implementation does nothing
			return;
		}
		
		// volume allocation/deallocation
		Object ***allocate () {
			return allocate(size);
		}
		
		Object ***allocate (const int sz[3]) {
			Object ***v = new Object**[sz[0]];
			for (int i = 0; i < sz[0]; ++i) {
				v[i] = new Object*[sz[1]];
				for (int j = 0; j < sz[1]; ++j) {
					v[i][j] = new Object[sz[2]];
				}
			}
			return v;
		}
		
		void deallocate (Object ***v) {
			deallocate(v, size);
		}
		
		void deallocate (Object ***v, const int sz[3]) {
			if (v == NULL) return;
			
			for (int i = 0; i < sz[0]; ++i) {
				for (int j = 0; j < sz[1]; ++j) {
					delete[] v[i][j];
				}
				delete[] v[i];
			}
			delete[] v;
		}
		
		// determine the 1D convolution mask
		double *compute1DGaussianConvolutionMask (const double sigma, int& cutOff) {
			cutOff = (int)round(fabs(3.0 * sigma));
			double *h = new double[2*cutOff + 1];
			if (sigma == 0.0) {
				h[0] = 1.0;
			} else {
				double sum = 0.0;
				for (int i = -cutOff; i <= cutOff; ++i) {
					h[i + cutOff] = std::exp(-0.5*i*i/(sigma*sigma));
					sum += h[i + cutOff];
				}
				// need to normalize
				sum = 1.0/sum;
				for (int i = -cutOff; i <= cutOff; ++i) {
					h[i + cutOff] *= sum;
				}
			}
			return h;
		}
		
		// smoothed is expected to be memory allocated
		// do not skip even when sigma = 0
		void compute1DGaussianSmoothing (const int dir, const double sigma, Object ***input, Object ***smoothed) {
			if (dir < 0 || dir > 2) {
				cerr << "invalid direction specification by dir = ";
				cerr << dir << endl;
				exit (1);
			}
			
			// storage for 1D convolution mask
			int cutOff = 0;
			double *h = compute1DGaussianConvolutionMask(sigma, cutOff);
			
			// smoothing loop along the direction specified by "dir"
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						smoothed[i][j][k] = zero;
						int idx[3] = {i, j, k};
						idx[dir] -= cutOff;
						for (int l = -cutOff; l <= cutOff; ++idx[dir], ++l) {
							if (idx[dir] < 0 || idx[dir] >= size[dir]) {
								smoothed[i][j][k] += h[l + cutOff] * bg;
							} else {
								smoothed[i][j][k] += h[l + cutOff] * input[idx[0]][idx[1]][idx[2]];
							}
						}
					}
				}
			}
			delete[] h;
		}
		
		// Gaussian Smoothing
		// sigma is in voxel spacing
		// smoothed is expected to be memory allocated
		void gaussianSmoothing (const double sigma[3], Object ***smoothed) {
			if (smoothed == NULL && sigma[0] == 0.0 && sigma[1] == 0.0 && sigma[2] == 0.0) {
				return;
			}
			
			cout << "Gaussian smoothing: sigma = [";
			cout << setprecision(3);
			cout << setw(4) << sigma[0] << ", ";
			cout << setw(4) << sigma[1] << ", ";
			cout << setw(4) << sigma[2];
			cout << "] ... " << flush;
			clock_t t1 = clock();
			
			// internal buffer volume
			Object ***buffer = allocate();
			
			// go through each direction
			if (smoothed != NULL) {
				// original data is not replaced
				compute1DGaussianSmoothing(0, sigma[0], voxel, smoothed);
				compute1DGaussianSmoothing(1, sigma[1], smoothed, buffer);
				compute1DGaussianSmoothing(2, sigma[2], buffer, smoothed);
			} else {
				// original data is replaced
				compute1DGaussianSmoothing(0, sigma[0], voxel, buffer);
				compute1DGaussianSmoothing(1, sigma[1], buffer, voxel);
				compute1DGaussianSmoothing(2, sigma[2], voxel, buffer);
			}
			
			// clear the memory
			if (smoothed != NULL) {
				deallocate(buffer);
			} else {
				deallocate(voxel);
				voxel = buffer;
			}
			
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
		}
		
		// input/output
		void initVTKReader (VTKReader& in) {
			in.getDimensions(size);
			in.getSpacing(vsize);
			in.getOrigin(origin);
		}
		
		void initVTKWriter (VTKWriter& out) {
			out.setDimensions(size);
			out.setSpacing(vsize);
			out.setOrigin(origin);
			out.writeHeader();
		}
		
		void initVTKSliceWriter (VTKWriter& out, int k) {
			int slSize[3] = {size[0], size[1], 1};
			out.setDimensions(slSize);
			out.setSpacing(vsize);
			double slOrigin[3] = {origin[0], origin[1], origin[2]};
			slOrigin[2] += vsize[3] * k;
			out.setOrigin(slOrigin);
		}
		
		bool readVol (VTKReader& in) {
			initVTKReader(in);
			if (voxel == NULL) {
				voxel = allocate();
			}
			
			// z first
			cout << "Reading the buffer ... " << flush;
			clock_t t1 = clock();
			for (int k = 0; k < size[2]; ++k) {
				if (!readSliceAux(in, k)) {
					return false;
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			return true;
		}
		
		bool readSliceAux (VTKReader& in, int k) {
			bool flag;
			
			// SHOULD HAVE x loop as the inner most then y loop
			for (int j = 0; j < size[1]; ++j) {
				for (int i = 0; i < size[0]; ++i) {
					flag = in.readElement(voxel[i][j][k]);
					if (!flag) {
						cerr << "Data reading failed" << endl;
						return false;
					}
				}
			}
			return true;
		}
		
		bool writeVTKVol (VTKWriter& out, Object ***vol) {
			initVTKWriter(out);
			
			// z first
			cout << "Writing the buffer ... " << flush;
			clock_t t1 = clock();
			for (int k = 0; k < size[2]; ++k) {
				if (!writeSliceAux(out, vol, k)) {
					return false;
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			
			return true;
		}
		
		bool writeSliceAux (VTKWriter& out, Object ***vol, int k) {
			bool flag;
			// SHOULD HAVE x loop as the inner most then y loop
			for (int j = 0; j < size[1]; ++j) {
				for (int i = 0; i < size[0]; ++i) {
					flag = out.writeElement(vol[i][j][k]);
					if (!flag) {
						cerr << "Data writing failed" << endl;
						return false;
					}
				}
			}
			return true;
		}
		
		public:
		
		Volume (const int sz[3], bool enableGrad = false) : VoxelSpace(sz) {
			voxel = allocate();
			if (enableGrad) {
				gradEnabled = true;
				for (int i = 0 ; i < 3; ++i) {
					grad[i] = allocate();
				}
			} else {
				gradEnabled = false;
				for (int i = 0; i < 3; ++i) {
					grad[i] = NULL;
				}
			}
		}
		
		Volume (const VoxelSpace& vs, bool enableGrad = false) : VoxelSpace(vs) {
			voxel = allocate();
			if (enableGrad) {
				gradEnabled = true;
				for (int i = 0 ; i < 3; ++i) {
					grad[i] = allocate();
				}
			} else {
				gradEnabled = false;
				for (int i = 0; i < 3; ++i) {
					grad[i] = NULL;
				}
			}
		}
		
		virtual ~Volume () {
			deallocate(voxel);
			clearGradient();
		}
		
		void fillWithBackground () {
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						voxel[i][j][k] = bg;
					}
				}
			}
		}
		
		void setBackground (const Object& in) {
			bg = in;
		}
		
		void getBackground (Object& out) const {
			out = bg;
		}
		
		// computing the gradient of voxel
		// 
		// the gradient are computed in the absolute
		// scale
		void buildGradient (const double *sigma = NULL) {
			if (!gradEnabled) {
				gradEnabled = true;
			}
			
			// allocate space if not done so
			if (grad[0] == NULL) {
				for (int i = 0; i < 3; ++i) {
					grad[i] = allocate();
				}
			}
			
			// if requested, create a smoothed copy of the image
			// for gradient computation
			// typically, image might have been smoothed beforehand
			// this will introduce additional smoothing that might
			// not be required or desired in most cases
			Object ***smooth = NULL;
			if (sigma != NULL && (sigma[0] != 0.0 || sigma[1] != 0.0 || sigma[2] != 0.0)) {
				smooth = allocate();
				gaussianSmoothing(sigma, smooth);
			} else {
				smooth = voxel;
			}
			
			Object *current = NULL;
			int index[3];
			double frac[3] = {0.5 / vsize[0], 0.5 / vsize[1], 0.5 / vsize[2]};
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						for (int m = 0; m < 3; ++m) {
							index[0] = i;
							index[1] = j;
							index[2] = k;
							current = &grad[m][i][j][k];
							*current = zero;
							if (index[m] == 0) {
								if (index[m] < size[m] - 1) {
									index[m] = 1;
									*current += smooth[index[0]][index[1]][index[2]];
								}
							} else if (index[m] == size[m] - 1) {
								if (index[m] > 0) {
									index[m] = size[m] - 2; 
									*current -= smooth[index[0]][index[1]][index[2]];
								}
							} else {
								index[m] += 1;
								*current += smooth[index[0]][index[1]][index[2]];
								index[m] -= 2;
								*current -= smooth[index[0]][index[1]][index[2]];
							}
							*current *= frac[m];
						}
					}
				}
			}
			
			// makes sure to clean up if required
			if (sigma != NULL) {
				deallocate(smooth);
			}

			return;
		}
		
		void clearGradient () {
			if (gradEnabled) {
				gradEnabled = false;
				for (int i = 0; i < 3; ++i) {
					deallocate(grad[i]);
				}
			}
		}
		
		void gaussianSmoothing (const double sigma = 1.0) {
			const double sigma3[3] = {sigma, sigma, sigma};
			gaussianSmoothing(sigma3, NULL);
		}
		
		void gaussianSmoothing (const double sigma[3]) {
			gaussianSmoothing(sigma, NULL);
		}
		
		void gaussianSmoothing2D (const int dir[2], const double sigma[2]) {
			// internal buffer volume
			Object ***buffer = allocate();
			// go through each direction
			compute1DGaussianSmoothing(dir[0], sigma[0], voxel, buffer);
			compute1DGaussianSmoothing(dir[1], sigma[1], buffer, voxel);
			deallocate(buffer);
		}
		
		// volume copy
		Volume<Object>& operator= (const Volume<Object>& rhs) {
			const int xsize = size[0];
			const int ysize = size[1];
			const int zsize = size[2];
			
			for (int i = 0; i < xsize; ++i) {
				for (int j = 0; j < ysize; ++j) {
					for (int k = 0; k < zsize; ++k) {
						voxel[i][j][k] = rhs.voxel[i][j][k];
					}
				}
			}
			
			return *this;
		}
		
		// volume addition (useful for creating averaged image)
		// two volumes should be of same size
		Volume<Object>& operator+= (const Volume<Object>& rhs) {
			cout << "Adding " << rhs.getName() << " to ";
			cout << this->name << " ... " << flush;
			clock_t t1 = clock();
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						voxel[i][j][k] += rhs.voxel[i][j][k];
					}
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "add"));
			return *this;
		}
		
		// volume difference
		// two volumes should be of same size
		Volume<Object>& operator-= (const Volume<Object>& rhs) {
			cout << "Subtracting " << rhs.getName() << " from ";
			cout << this->name << " ... " << flush;
			clock_t t1 = clock();
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						voxel[i][j][k] -= rhs.voxel[i][j][k];
					}
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "subtract"));
			return *this;
		}
		
		// volume scalar multiplication
		Volume<Object>& operator*= (double rhs) {
			cout << "Voxelwise scaling " << this->name << " by ";
			cout << rhs << " ... " << flush;
			clock_t t1 = clock();
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						voxel[i][j][k] *= rhs;
					}
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "scale"));
			return *this;
		}
		
		// voxel-wise division
		Volume<Object>& divide (const Volume<double>& rhs, const double tiny = 1.0) {
			cout << "Dividing " << this->name << " by ";
			cout << rhs.getName() << " ... " << flush;
			clock_t t1 = clock();
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					for (int k = 0; k < size[2]; ++k) {
						if (rhs.voxel[i][j][k] < tiny) {
							voxel[i][j][k] = zero;
						} else {
							voxel[i][j][k] /= rhs.voxel[i][j][k];
						}
					}
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "divide"));
			return *this;
		}
		
		// build mosaic-like composition among corresponding
		// slices of two volumes
		Volume<Object>& mosaicCompositionWith (const Volume<Object>& rhs, int mosaic) {
			cout << "Mosaicing " << this->name << " with ";
			cout << rhs.getName() << " ... " << flush;
			clock_t t1 = clock();
			for (int i = 0; i < size[0]; ++i) {
				for (int j = 0; j < size[1]; ++j) {
					if ((i/mosaic)%2 == (j/mosaic)%2) {
						for (int k = 0; k < size[2]; ++k) {
							voxel[i][j][k] = rhs.voxel[i][j][k];
						}
					}
				}
			}
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "mosaic"));
			return *this;
		}
		
		// get the value of gradEnabled
		bool getGradEnabled () const {
			return gradEnabled;
		}
		
		// the backward resampling function
		bool getVoxelAt (const Vector3D& vec, Object& out, const int intp = 0) const {
			// more efficient to use local variables 
			// macro
			if (intp != 0) {
				// nearest neighbour interpolation
				Vector3D loc(vec);
				toRel(loc);
				for (int i = 0; i < 3; ++i) {
					// because we use "rounding" rather than "flooring"
					if (loc[i] > -0.5 && loc[i] < size[i] - 0.5) {
						continue;
					} else {
						out = bg;
						return false;
					}
				}
				out = voxel[(int)round(loc[0])][(int)round(loc[1])][(int)round(loc[2])];
				return true;
			}
			// otherwise
			_COMMON_VAR
			if (computeBottomLeftCornerIndexAndLambdaRegion(vec, bottomLeft, lambda)) {
				computeCornerIndices(bottomLeft, cornerIndex);
				computeCornerObjects(cornerIndex, corner);
				out = interpolate8(corner, lambda);
				return true;
			} else {
				out = bg;
				return false;
			}
		}
		
		bool getVoxelAt (const Vector3D& vec, Object& out, Object gradOut[3], const int intp = 0) const {
			// macro
			_COMMON_VAR
			if (computeBottomLeftCornerIndexAndLambdaRegion(vec, bottomLeft, lambda)) {
				computeCornerIndices(bottomLeft, cornerIndex);
				computeCornerObjects(cornerIndex, corner);
				switch (intp) {
					default:
					case 0: out = interpolate8(corner, lambda);
						   break;
				}
				
				for (int i = 0; i < 3; ++i) {
					computeGradCornerObjects(i, cornerIndex, corner);
					switch (intp) {
						default:
						case 0: gradOut[i] = interpolate8(corner, lambda);
							   break;
					}
				}
				
				return true;
			} else {
				out = bg;
				
				// when in the background
				// no need to compute gradient
				
				return false;
			}
		}
		
		bool putVoxelAt (const Vector3D& vec, const Object& in) {
			Vector3D loc(vec);
			toRel(loc);
			for (int i = 0; i < 3; ++i) {
				if (loc[i] >=0 && loc[i] < size[i]) {
					continue;
				} else {
					return false;
				}
			}
			voxel[(int)round(loc[0])][(int)round(loc[1])][(int)round(loc[2])] = in;
			return true;
		}
		
    // voxel replacement with reflection 
    Volume<Object>& replacementWithReflection (const Volume<double>* mask, const Reflection3D& trans) {
      cout << "Replacing the voxels in " << mask->getName();
      cout << " with the voxels of its mirror reflection about midsaggital ";
      cout << " ... " << flush;
      clock_t t1 = clock();
      for (int i = 0; i < size[0]; ++i) {
        for (int j = 0; j < size[1]; ++j) {
          for (int k = 0; k < size[2]; ++k) {
            if (mask->voxel[i][j][k] == 0) {
              continue;
            }
            Vector3D vec(i, j, k);
            this->toAbs(vec);
            vec *= trans;
            getVoxelAt(vec, voxel[i][j][k]);
          }
        }
      }
      // some code for testing
      /*for (int i = 0; i < size[0]; ++i) {
        for (int j = 0; j < size[1]; ++j) {
          for (int k = 0; k < size[2]; ++k) {
            if (mask->voxel[i][j][k] == 0) {
              continue;
            }
            Vector3D vec(i, j, k);
            this->toAbs(vec);
            vec *= trans;
            putVoxelAt(vec, 0);
          }
        }
      }*/
      clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			this->setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(), "replacement"));
			return *this;
    }
		
		// the forward resampling function
		bool putVoxelAt (const Vector3D& vec, const Object& in, Volume<double>& normalize) {
			int cornerIndex[2][2][2][3];
			double lambda[3];
			int bottomLeft[3];
			Object *corner[2][2][2];
			double *corner2[2][2][2];
			if (computeBottomLeftCornerIndexAndLambdaRegion(vec, bottomLeft, lambda)) {
				computeCornerIndices(bottomLeft, cornerIndex);
				computeCornerObjects(voxel, cornerIndex, corner, normalize, corner2);
				distribute8(in, corner, corner2, lambda);
				return true;
			} else {
				return false;
			}
		}
		
		// Compute the corner objects of an interpolation cube for
		// the input volume
		// 
		// input: voxel (the input volume)
		//        cornerIndex (indices of all corners)
		// output: corner
		void computeCornerObjects (Object ***voxel, const int cornerIndex[2][2][2][3],
			const Object *corner[2][2][2]) const {
			int tmp;
			int index[3];
			bool beBg;
			
			// loop through cube corners z,y,x
			for (int k = 0; k < 2; ++k) {
				for (int j = 0; j < 2; ++j) {
					for (int i = 0; i < 2; ++i) {
						beBg = false;
					 	// loop through X, Y, Z directions
						for (int m = 0; m < 3; ++m) {
							tmp = cornerIndex[k][j][i][m];
							if (tmp < 0 || tmp >= size[m]) {
								// should be bg Object
								beBg = true;
								break;
							} else {
								index[m] = tmp;
							}
						}
						
						if (beBg) {
							corner[k][j][i] = &bg;
						} else {
							corner[k][j][i] = &voxel[index[0]][index[1]][index[2]];
						}
					}
				}
			}
			
			return;
		}
		
		// compute the corner objects of an interpolation cube for 
		// the volume
		void computeCornerObjects (const int cornerIndex[2][2][2][3],
			const Object *corner[2][2][2]) const {
			return computeCornerObjects(voxel, cornerIndex, corner);
		}
		
		void computeCornerObjects (Object ***voxel, const int cornerIndex[2][2][2][3],
			Object *corner[2][2][2], Volume<double>& auxi, double *corner2[2][2][2]) const {
			int tmp;
			int index[3];
			bool beBg;
			
			// loop through cube corners z,y,x
			for (int k = 0; k < 2; ++k) {
				for (int j = 0; j < 2; ++j) {
					for (int i = 0; i < 2; ++i) {
						beBg = false;
					 	// loop through X, Y, Z directions
						for (int m = 0; m < 3; ++m) {
							tmp = cornerIndex[k][j][i][m];
							if (tmp < 0 || tmp >= size[m]) {
								// should be bg Object
								beBg = true;
								break;
							} else {
								index[m] = tmp;
							}
						}
						
						if (beBg) {
							// different treatment here
							corner[k][j][i] = NULL;
							corner2[k][j][i] = NULL;
						} else {
							corner[k][j][i] = &voxel[index[0]][index[1]][index[2]];
							corner2[k][j][i] = &auxi.voxel[index[0]][index[1]][index[2]];
						}
					}
				}
			}
			
			return;
		}

		// compute the corner objects of an interpolation cube for 
		// the gradient of the volume
		// 
		// the additional input : the gradient component to be computed
		// 
		// assuming gradient is available
		void computeGradCornerObjects (const int m, const int cornerIndex[2][2][2][3],
			const Object *corner[2][2][2]) const {
			return computeCornerObjects(grad[m], cornerIndex, corner);
		}
		
		void distribute8 (const Object& in, Object *corner[2][2][2], double *corner2[2][2][2], const double lambda[3]) {
			const double mult[3][2] = {
				{1.0 - lambda[0], lambda[0]},
				{1.0 - lambda[1], lambda[1]},
				{1.0 - lambda[2], lambda[2]},
			};
			
			int idx[3];
			// indexing z y x
			for (idx[2] = 0; idx[2] < 2; ++idx[2]) {
				for (idx[1] = 0; idx[1] < 2; ++idx[1]) {
					for (idx[0] = 0; idx[0] < 2; ++idx[0]) {
						double coeff = 1.0;
						for (int m = 0; m < 3; ++m) {
							coeff *= mult[m][idx[m]];
						}
						if (corner[idx[2]][idx[1]][idx[0]] != 0) {
							*corner[idx[2]][idx[1]][idx[0]] += coeff * in;
							*corner2[idx[2]][idx[1]][idx[0]] += coeff;
						}
					}
				}
			}
		}
		
		Object interpolate8 (
			const Object *corner[2][2][2], const double lambda[3]) const {
			return interpolate8(
				*corner[0][0][0], *corner[0][0][1],
				*corner[0][1][0], *corner[0][1][1],
				*corner[1][0][0], *corner[1][0][1],
				*corner[1][1][0], *corner[1][1][1],
				lambda);
		}
		
		Object interpolate8 (
			const Object& c000, const Object& c001,
			const Object& c010, const Object& c011,
			const Object& c100, const Object& c101,
			const Object& c110, const Object& c111, const double lambda[3]) const {
			return interpolate2(
					interpolate4(c000, c001, c010, c011, lambda),
					interpolate4(c100, c101, c110, c111, lambda),
					lambda[2]);
		}
		
		Object interpolate4 (
			const Object& c00, const Object& c01,
			const Object& c10, const Object& c11, const double lambda[2]) const {
			return interpolate2(
						interpolate2(c00, c01, lambda[0]),
						interpolate2(c10, c11, lambda[0]),
						lambda[1]);
		}
		
		Object interpolate2 (
			const Object& c0, const Object& c1, const double lambda) const {
			Object tmp(c0);
			tmp *= 1 - lambda;
			tmp += lambda * c1;
			return tmp;
		}
		
		void changeOrientation (const char *input, const char *output) {
			// if the input and output orientations are the same
			// do nothing
			if (strcmp(input, output) == 0) {
				cout << "the input and output orientations are the same.  nothing to do." << endl;
				return;
			}
			cout << "change the volume from " << input;
			cout << " to " << output << " ... " << flush;
			clock_t t1 = clock();
			
			// compute the orientation mapping
			int mapping[3];
			bool dir[3];
			VoxelSpace::computeOrientationMapping(input, output, mapping, dir);
			
			// back up the old volume data and set up the new volume
			Object ***old_voxel = voxel;
			int old_size[3];
			getSize(old_size);
			// remap the voxel space
			remap(mapping);
			voxel = allocate(size);
			// current indexing of the volume
			int oidx[3];
			// new indexing of the volume
			int nidx[3];
			for (oidx[0] = 0; oidx[0] < old_size[0]; ++oidx[0]) {
				for (oidx[1] = 0; oidx[1] < old_size[1]; ++oidx[1]) {
					for (oidx[2] = 0; oidx[2] < old_size[2]; ++oidx[2]) {
						for (int i = 0; i < 3; ++i) {
							if (dir[i]) {
								nidx[mapping[i]] = oidx[i];
							} else {
								nidx[mapping[i]] = old_size[i] - 1 - oidx[i];
							}
						}
						voxel[nidx[0]][nidx[1]][nidx[2]] =
							old_voxel[oidx[0]][oidx[1]][oidx[2]];
					}
				}
			}
			// clear the old volume data
			deallocate(old_voxel, old_size);
			// object-specific orientation mapping
			objectSpecificOrientationMapping(mapping, dir);
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			setName(getFilenameForSymTensor3DVolumeDerived(this->name.c_str(),output));
		}
		
		void extractRegionOf (const Volume<Object>& vol, const char *align) {
			Vector3D vec;
			const int xsize = size[0];
			const int ysize = size[1];
			const int zsize = size[2];
			
			Vector3D shift;
			if (strcmp("center", align) == 0) {
				cout << "matching the center of the old and the new voxel spaces" << endl;
				for (int i = 0; i < 3; ++i) {
					shift[i] = 0.5 * (size[i] - 1) * vsize[i];
					shift[i] -= 0.5 * (vol.size[i] - 1) * vol.vsize[i];
				}
			} else if (strcmp("origin", align) == 0) {
				cout << "matching the origin of the old and the new voxel spaces" << endl;
			}
			
			for (int i = 0; i < xsize; ++i) {
				for (int j = 0; j < ysize; ++j) {
					for (int k = 0; k < zsize; ++k) {
						vec[0] = i;
						vec[1] = j;
						vec[2] = k;
						toAbs(vec);
						vec -= shift;
						vol.getVoxelAt(vec, voxel[i][j][k]);
					}
				}
			}
			
			return;
		}
		
		bool writeVectorialNifti (const char *filename, const int dim, const int intent_code) {
			// set up nifti image structure
			nifti_image *nim = toNifti(filename, dim, intent_code);
			
			//cout << "Converting the buffer ... " << flush;
			clock_t t1 = clock();
			convertVectorialVoxelToNiftiData<float>(nim, dim);
			clock_t t2 = clock();
			//cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			
			// write nifti and clean up
			cout << "Writing " << filename <<  " ... " << flush;
			t1 = clock();
			nifti_image_write(nim);
			t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			nifti_image_free(nim);
			return true;
		}
		
		bool writeScalarNifti (const char *filename, const int intent_code = 0) {
			// set up nifti image structure
			nifti_image *nim = toNifti(filename, 1, intent_code);
			
			//cout << "Converting the buffer ... " << flush;
			clock_t t1 = clock();
			convertScalarVoxelToNiftiData<float>(nim);
			clock_t t2 = clock();
			//cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			
			// write nifti and clean up
			cout << "Writing " << filename << " ... " << flush;
			t1 = clock();
			nifti_image_write(nim);
			t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			nifti_image_free(nim);
			return true;
		}
		
		// map vectorial-valued volume to nifti data
		template <class PixelType>
		void convertVectorialVoxelToNiftiData (nifti_image *nim, const int dim) {
			// copy the internal buffer to output buffer
			int index = 0;
			for (int m = 0; m < dim; ++m) {
				for (int k = 0; k < size[2]; ++k) {
					for (int j = 0; j < size[1]; ++j) {
						for (int i = 0; i < size[0]; ++i) {
							((PixelType *)(nim->data))[index] = (PixelType) voxel[i][j][k][m];
							++index;
						}
					}
				}
			}
		}
		
		// map scalar-valued volume to nifti data
		template <class PixelType>
		void convertScalarVoxelToNiftiData (nifti_image *nim) {
			// copy the internal buffer to output buffer
			int index = 0;
			for (int k = 0; k < size[2]; ++k) {
				for (int j = 0; j < size[1]; ++j) {
					for (int i = 0; i < size[0]; ++i) {
						((PixelType *)(nim->data))[index] = (PixelType) voxel[i][j][k];
						++index;
					}
				}
			}
		}
		
		// map nifti data to vector-valued volume
		// including tensor data stored in vectorial format internally
		template <class PixelType>
		void convertNiftiDataToVectorialVoxel (const nifti_image *nim, const int dim) {
			// first determine orientation mapping
			int mapping[3];
			bool dir[3];
			convertToLPI(nim, mapping, dir);
			// remember the input size
			int isize[3];
			getSize(isize);
			// remap the voxel space information if necessary
			remap(mapping);
			// allocate the memory
			voxel = allocate(size);
			// input indexing of the volume
			int iidx[3];
			// native indexing of the volume
			int nidx[3];
			// copy values
			// note that nifti_image_read does not scale images
			int index = 0;
			// the ordering of the loops is critical for correctness
			for (int m = 0; m < dim; ++m) {
				for (iidx[2] = 0; iidx[2] < isize[2]; ++iidx[2]) {
					if (dir[2]) {
						nidx[mapping[2]] = iidx[2];
					} else {
						nidx[mapping[2]] = isize[2] - 1 - iidx[2];
					}
					for (iidx[1] = 0; iidx[1] < isize[1]; ++iidx[1]) {
						if (dir[1]) {
							nidx[mapping[1]] = iidx[1];
						} else {
							nidx[mapping[1]] = isize[1] - 1 - iidx[1];
						}
						for (iidx[0] = 0; iidx[0] < isize[0]; ++iidx[0]) {
							if (dir[0]) {
								nidx[mapping[0]] = iidx[0];
							} else {
								nidx[mapping[0]] = isize[0] - 1 - iidx[0];
							}
							if (nim->scl_slope == 0) {
								voxel[nidx[0]][nidx[1]][nidx[2]][m] = ((PixelType *)(nim->data))[index];
							} else {
								voxel[nidx[0]][nidx[1]][nidx[2]][m] = ((PixelType *)(nim->data))[index] * nim->scl_slope + nim->scl_inter;
							}
							++index;
						}
					}
				}
			}
			// now need to remap voxelwise data to account for orientation changes
			objectSpecificOrientationMapping(mapping, dir);
			return;
		}
		
		// map nifti data to scalar-valued volume
		template <class PixelType>
		void convertNiftiDataToScalarVoxel (const nifti_image *nim) {
			// first determine orientation mapping
			int mapping[3];
			bool dir[3];
			convertToLPI(nim, mapping, dir);
			// remember the input size
			int isize[3];
			getSize(isize);
			// remap the voxel space information if necessary
			remap(mapping);
			// allocate the memory
			voxel = allocate(size);
			// input indexing of the volume
			int iidx[3];
			// native indexing of the volume
			int nidx[3];
			// copy values
			// note that nifti_image_read does not scale images
			int index = 0;
			// the ordering of the loops is critical for correctness
			for (iidx[2] = 0; iidx[2] < isize[2]; ++iidx[2]) {
				if (dir[2]) {
					nidx[mapping[2]] = iidx[2];
				} else {
					nidx[mapping[2]] = isize[2] - 1 - iidx[2];
				}
				for (iidx[1] = 0; iidx[1] < isize[1]; ++iidx[1]) {
					if (dir[1]) {
						nidx[mapping[1]] = iidx[1];
					} else {
						nidx[mapping[1]] = isize[1] - 1 - iidx[1];
					}
					for (iidx[0] = 0; iidx[0] < isize[0]; ++iidx[0]) {
						if (dir[0]) {
							nidx[mapping[0]] = iidx[0];
						} else {
							nidx[mapping[0]] = isize[0] - 1 - iidx[0];
						}
						if (nim->scl_slope == 0) {
							voxel[nidx[0]][nidx[1]][nidx[2]] = ((PixelType *)(nim->data))[index];
						} else {
							voxel[nidx[0]][nidx[1]][nidx[2]] = ((PixelType *)(nim->data))[index] * nim->scl_slope + nim->scl_inter;
						}
						++index;
					}
				}
			}
			return;
		}
		
		bool readVectorialNifti (const char *filename, const int dim, const int intent_code) {
			// common nifti read stuff
			nifti_image *nim = readNiftiCommon(filename, dim, intent_code);
			if (nim == NULL) {
				return false;
			}
			
			// copy values
			//cout << "Converting the buffer ... " << flush;
			//clock_t t1 = clock();
			switch (nim->datatype) {
				case DT_UINT8:
					convertNiftiDataToVectorialVoxel<unsigned char>(nim, dim);
					break;
				case DT_INT8:
					convertNiftiDataToVectorialVoxel<char>(nim, dim);
					break;
				case DT_UINT16:
					convertNiftiDataToVectorialVoxel<unsigned short>(nim, dim);
					break;
				case DT_INT16:
					convertNiftiDataToVectorialVoxel<short>(nim, dim);
					break;
				case DT_UINT32:
					convertNiftiDataToVectorialVoxel<unsigned int>(nim, dim);
					break;
				case DT_INT32:
					convertNiftiDataToVectorialVoxel<int>(nim, dim);
					break;
				case DT_UINT64:
					convertNiftiDataToVectorialVoxel<unsigned long>(nim, dim);
					break;
				case DT_INT64:
					convertNiftiDataToVectorialVoxel<long>(nim, dim);
					break;
				case DT_FLOAT32:
					convertNiftiDataToVectorialVoxel<float>(nim, dim);
					break;
				case DT_FLOAT64:
					convertNiftiDataToVectorialVoxel<double>(nim, dim);
					break;
				case DT_FLOAT128:
					convertNiftiDataToVectorialVoxel<long double>(nim, dim);
					break;
				default:
					cerr << "Unsupported datatype : " << nim->datatype << endl;
					exit (1);
			}
			//clock_t t2 = clock();
			//cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			
			// clear memory
			nifti_image_free(nim);
			
			return true;
		}
		
		bool readScalarNifti (const char *filename, const int intent_code = 0) {
			// common nifti read stuff
			nifti_image *nim = readNiftiCommon(filename, 1, intent_code);
			if (nim == NULL) {
				return false;
			}
			
			// copy values
			//cout << "Converting the buffer ... " << flush;
			//clock_t t1 = clock();
			switch (nim->datatype) {
				case DT_UINT8:
					convertNiftiDataToScalarVoxel<unsigned char>(nim);
					break;
				case DT_INT8:
					convertNiftiDataToScalarVoxel<char>(nim);
					break;
				case DT_UINT16:
					convertNiftiDataToScalarVoxel<unsigned short>(nim);
					break;
				case DT_INT16:
					convertNiftiDataToScalarVoxel<short>(nim);
					break;
				case DT_UINT32:
					convertNiftiDataToScalarVoxel<unsigned int>(nim);
					break;
				case DT_INT32:
					convertNiftiDataToScalarVoxel<int>(nim);
					break;
				case DT_UINT64:
					convertNiftiDataToScalarVoxel<unsigned long>(nim);
					break;
				case DT_INT64:
					convertNiftiDataToScalarVoxel<long>(nim);
					break;
				case DT_FLOAT32:
					convertNiftiDataToScalarVoxel<float>(nim);
					break;
				case DT_FLOAT64:
					convertNiftiDataToScalarVoxel<double>(nim);
					break;
				case DT_FLOAT128:
					convertNiftiDataToScalarVoxel<long double>(nim);
					break;
				default:
					cerr << "Unsupported datatype : " << nim->datatype << endl;
					exit (1);
			}
			//clock_t t2 = clock();
			//cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			
			// clear memory
			nifti_image_free(nim);
			
			return true;
		}
		
		nifti_image *readNiftiCommon (const char *filename, const int dim, const int intent_code) {
			// read nifti file
			cout << "Reading " << filename << " ... " << flush;
			clock_t t1 = clock();
			nifti_image *nim = fromNifti(filename, dim, intent_code);
			clock_t t2 = clock();
			cout << "Done in " << (t2 - t1)/(double)CLOCKS_PER_SEC << 's' << endl;
			return nim;
		}
		
		// center of mass
		virtual Vector3D computeCenterOfMass () const {
			return Vector3D();
		}
		
		// moment of inertia
		virtual SymTensor3D computeMomentOfInertia () const {
			return SymTensor3D();
		}
		
		// logarithm
		virtual void log (const bool force = true) {};
		// exponetial
		virtual void exp () {};
	};
	
}

#endif

