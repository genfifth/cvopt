#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, importlib


argvs = sys.argv

if (len(argvs) != 4):
	 print("this module need 3 arguments.")
	 quit()

tgt_type = argvs[1]
module_rootdir = argvs[2]
module_name = argvs[3]

if tgt_type == "module":
	sys.path.append(module_rootdir)
	module = importlib.import_module(module_name)
	ret = [i for i in dir(module) if i[0] != "_"]
elif tgt_type == "dir":
	ret = []
	for d in os.listdir(os.path.join(module_rootdir, module_name)):
		if os.path.isdir(os.path.join(module_rootdir, module_name, d)): 
			if d[0] != "_":
				ret.append(d)
else:
	 print("tgt_type(arg[1]) must be module or dir.")
	 quit()

print("\n".join(ret))
quit()