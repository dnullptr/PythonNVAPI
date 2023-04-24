import pynvraw

'''NVAPI Py3 implementation usage for diagnosis and overclocking using the CLI.
   Daniel Ohayon @dnullptr 2023
   Use at your own risk. Can brick your GPU or burn your house down.

   Make sure you have the latest drivers installed and that you have the latest CUDA toolkit installed.'''
   
# init all the pointers and stuff
api = pynvraw.NvAPI() # init api
gpus = pynvraw.get_gpus() # get gpus
nv_physical = api.gpu_handles[0] # get nv physical handle
my_gpu_struct = pynvraw.Gpu(nv_physical,api) # get gpu struct
clocks = pynvraw.Clocks(core=150,memory=200,processor=0,video=0) # Prepare clocks for OC (BEWARE!)

# driver testing and info gathering
print(api.get_memory_info(nv_physical)) #meminfo object using nv physical handle
print(api.get_ram_type(nv_physical)) #ram type
print(my_gpu_struct.get_rail_powers()) #gpu rail powers
print(my_gpu_struct.name) #gpu name

# overclocking
print(my_gpu_struct.core_temp) #gpu temp in celsius
#my_gpu_struct.set_overclock(clocks) #GPU OC to 150 core and 200 memory
print(my_gpu_struct.get_overclock()) #will show the current OC
print(my_gpu_struct.fan) #gpu fan
