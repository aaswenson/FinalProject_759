// Allocate memory for ptrac data
particleTrack AllocatePtracData(const particleTrack hdata){
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
twoDmesh AllocateMeshData(const twoDmesh hmesh){
	cudaError_t error;
	twoDmesh dmesh = hmesh;
	int sizeI = hmesh.NI * sizeof(float);
	int sizeJ = hmesh.NJ * sizeof(float);
	int sizeK = hmesh.NK * sizeof(float);
	int sizeflux = hmesh.NI*hmesh.NJ*hmesh.NK*sizeof(float);

    error = cudaMalloc((void**)&dmesh.x, sizeI);
	error = cudaMalloc((void**)&dmesh.y, sizeJ);
	error = cudaMalloc((void**)&dmesh.z, sizeK);
	error = cudaMalloc((void**)&dmesh.flux, sizeflux);
	
    if (error != cudaSuccess)
	{
		printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
	return dmesh;
}

// copy the ptrac and mesh data to the device
void CopyDatatoDevice(particleTrack data, const particleTrack hdata,
                      twoDmesh dmesh, const twoDmesh hmesh)
{
	int size = hdata.Ntracks * sizeof(float);
	int sizeI = hmesh.NI * sizeof(float);
	int sizeJ = hmesh.NJ * sizeof(float);
	int sizeK = hmesh.NK * sizeof(float);
	int sizeflux = hmesh.NI*hmesh.NJ*hmesh.NK*sizeof(float);
	
    // ptrac data
    data.x_pos = hdata.x_pos;
	data.y_pos = hdata.y_pos;
	data.z_pos = hdata.z_pos;
	data.u = hdata.u;
	data.v = hdata.v;
	data.w = hdata.w;
	data.track_length = hdata.track_length;
    
    // mesh data
    dmesh.x = hmesh.x; dmesh.y = hmesh.y; dmesh.z = hmesh.z;
    dmesh.flux = hmesh.flux;

    // copy all data to device
    // Ptrac Data
	cudaMemcpy(data.x_pos, hdata.x_pos, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.y_pos, hdata.y_pos, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.z_pos, hdata.z_pos, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.u, hdata.u, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.v, hdata.v, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.w, hdata.w, size, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(data.track_length, hdata.track_length, size, 
			cudaMemcpyHostToDevice);
    // Mesh Data
	cudaMemcpy(dmesh.x, hmesh.x, sizeI, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.y, hmesh.y, sizeJ, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.z, hmesh.z, sizeK, 
			cudaMemcpyHostToDevice);
	cudaMemcpy(dmesh.flux, hmesh.flux, sizeflux, 
			cudaMemcpyHostToDevice);
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

