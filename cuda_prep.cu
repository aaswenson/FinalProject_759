
#include <cuda.h>

// Allocate memory for ptrac data
particleTrack AllocatePtracData(particleTrack hdata){
	cudaError_t error;
	particleTrack data = hdata;
	int size = hdata.Ntracks * sizeof(float);
	error = cudaMalloc((void**)&data.x_pos, size);
	error = cudaMalloc((void**)&data.y_pos, size);
	error = cudaMalloc((void**)&data.z_pos, size);
	error = cudaMalloc((void**)&data.u, size);
	error = cudaMalloc((void**)&data.v, size);
	error = cudaMalloc((void**)&data.w, size);
	error = cudaMalloc((void**)&data.track_length, size);
	if (error != cudaSuccess)
	{
		printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
	return data;
}

// Allocate memory for mesh data
twoDmesh AllocateMeshData(twoDmesh hmesh){
	cudaError_t error;
	twoDmesh dmesh = hmesh;
	int sizeI = hmesh.NI * sizeof(float);
	int sizeJ = hmesh.NJ * sizeof(float);
	int sizeK = hmesh.NK * sizeof(float);
	int sizeflux = hmesh.NI*hmesh.NJ*hmesh.NK*sizeof(float);

    error = cudaMalloc((void**)&dmesh.x, sizeI);
	cudaMalloc((void**)&dmesh.y, sizeJ);
	cudaMalloc((void**)&dmesh.z, sizeK);
	cudaMalloc((void**)&dmesh.flux, sizeflux);
	
    if (error != cudaSuccess)
	{
		printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
	return dmesh;
}

// copy the ptrac and mesh data to the device
void CopyDatatoDevice(particleTrack ddata, particleTrack hdata,
                      twoDmesh dmesh, twoDmesh hmesh)
{
    cudaError_t error;
	unsigned int size = hdata.Ntracks * sizeof(float);
    unsigned int meshsize = hmesh.NI * sizeof(float);
	unsigned int sizeflux = hmesh.NI*hmesh.NJ*hmesh.NK*sizeof(float);
	

    // copy all data to device
    // Ptrac Data
	error = cudaMemcpy(ddata.x_pos, hdata.x_pos, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.y_pos, hdata.y_pos, size, 
	    cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.z_pos, hdata.z_pos, size, 
	    cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.u, hdata.u, size, 
	    cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.v, hdata.v, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.w, hdata.w, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(ddata.track_length, hdata.track_length, size, 
			cudaMemcpyHostToDevice);
    // Mesh Data
    cudaMemcpy(dmesh.x, hmesh.x, meshsize, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.y, hmesh.y, meshsize, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.z, hmesh.z, meshsize, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.flux, hmesh.flux, sizeflux, 
			cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
	{
		printf("cudaMemcpy returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

}


//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps,float * error)
{
	for(unsigned int i = 0; i < elements; i++){
		float temp = sqrt((A[i]-B[i])*(A[i]-B[i]));
		*error+=temp;
		if(temp>eps){
			return false;
		} 
	}
	return true;
}

