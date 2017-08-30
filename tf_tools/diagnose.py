import gpustat
import resource
import objgraph


def memory_usage(device_num=0):
        g = gpustat.GPUStatCollection.new_query();
        k = g.gpus.keys()[device_num]
        return { 'device': int(g.gpus[k].entry['memory.used']),
                 'host': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def most_common_types(diff_from=None):
	c = {k: v for k, v in objgraph.most_common_types()}
	if diff_from is not None:
		for k, v in diff_from.iteritems():
			if k in c:
				c[k] -= diff_from[k]
			else:
				c[k] = -diff_from[k]
	return c
