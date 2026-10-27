import netCDF4 as nc
import subprocess
import shlex

ds = nc.Dataset("/home/evin/muTopOpt/tester/test_values2/batch_00_seed2/result2D_000.nc")
cmd = ds.getncattr("command_line")
# args = shlex.split(cmd)                      //uncomment these two lines to run the command 
# subprocess.run(["python"] + args)
print(cmd)