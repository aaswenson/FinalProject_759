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
	int size = (hmesh.N+1) * sizeof(float);
	int sizeflux = hmesh.N*hmesh.N*hmesh.N*sizeof(float);

    error = cudaMalloc((void**)&dmesh.x, size);
	cudaMalloc((void**)&dmesh.y, size);
	cudaMalloc((void**)&dmesh.z, size);
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
    unsigned int meshsize = (hmesh.N+1) * sizeof(float);
	unsigned int sizeflux = hmesh.N*hmesh.N*hmesh.N*sizeof(float);
	

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

void free_dev_mem(twoDmesh dmesh, particleTrack ddata){
    cudaFree(dmesh.flux);
    cudaFree(dmesh.x);
    cudaFree(dmesh.y);
    cudaFree(dmesh.z);
    cudaFree(ddata.x_pos);
    cudaFree(ddata.y_pos);
    cudaFree(ddata.z_pos);
    cudaFree(ddata.u);
    cudaFree(ddata.v);
    cudaFree(ddata.w);
    cudaFree(ddata.track_length);
    cudaFree(dmesh.flux);
}


//compare the data stored in two arrays on the host
float CompareResults(float* A, float* B, int elements, int Np)
{
	float sum=0;
    float error=0;
    for(unsigned int i = 0; i < elements; i++){
		error += A[i] - B[i];
        sum += A[i];
    }
     
    return sum;
}

